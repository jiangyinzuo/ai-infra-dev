"""以 bytecode 的符号指令改写 CPython 3.12–3.14 代码，不改写源码。

探针直接调用 dispatch；调用协议由当前解释器编译的微型模板确定。
只替换已知 NULL 调用协议，with/解释器内部的隐式 receiver 调用保持原样。
无法证明安全的操作保留原指令，并写入 unsupported，绝不捕获异常后重跑。
普通函数调用保留原 CALL 与调用帧，只有 runtime 明确识别的内建容器操作
才进入 dispatch；判定仅查看 callable，不重新求值用户表达式。
调用者负责只改写目标项目代码，并在执行前安装 runtime 的过滤与监控。
"""
from __future__ import annotations

import os
from types import CodeType, FunctionType
from typing import Callable, Literal, cast, overload

from bytecode import Bytecode, CellVar, FreeVar, Instr, Label, SetLineno, TryBegin, TryEnd
from bytecode.instr import InstrArg

from .protocol import OriginSpec, Site

Dispatch = Callable[[str, Site, tuple[object, ...]], object]
BytecodeEntry = Instr | Label | SetLineno | TryBegin | TryEnd

unsupported: list[tuple[str, str, int, str]] = []


def _call_template() -> tuple[bool, list[Instr]]:
    # 3.13 调换 callable/NULL 顺序；由编译器给出布局，避免硬编码版本表。
    namespace: dict[str, object] = {}
    exec(compile("def template(f, a, b, c):\n return f(a, b, c)\n", "<object-flow-call>", "exec"), namespace)
    template = cast(FunctionType, namespace["template"])
    instructions = [i for i in Bytecode.from_code(template.__code__) if isinstance(i, Instr)]
    null_index = next(n for n, i in enumerate(instructions) if i.name == "PUSH_NULL")
    function_index = next(n for n, i in enumerate(instructions)
                          if i.name.startswith("LOAD_FAST")
                          and (i.arg == "f" or isinstance(i.arg, tuple) and i.arg[0] == "f"))
    call_index = next(n for n, i in enumerate(instructions) if i.name == "CALL")
    start = call_index - 1 if instructions[call_index - 1].name == "PRECALL" else call_index
    return null_index < function_index, instructions[start:call_index + 1]


_NULL_FIRST, _CALL = _call_template()
_NULL = object()
_VALUE = object()
_FRAME = object()
_FRAME_NAMES = {"super", "locals", "globals", "vars", "eval", "exec", "_getframe", "__build_class__"}
_CALL_NAMES = {"CALL", "CALL_KW", "CALL_FUNCTION_EX"}
_BINDING_NAMES = {"STORE_FAST", "STORE_DEREF", "STORE_NAME", "STORE_GLOBAL"}
_BUILD_NAMES = {"BUILD_LIST", "BUILD_TUPLE", "BUILD_SET", "BUILD_MAP", "BUILD_CONST_KEY_MAP"}
_SUBSCRIPT_NAMES = {"BINARY_SUBSCR", "STORE_SUBSCR", "BINARY_SLICE", "STORE_SLICE"}
_COMPREHENSION_NAMES = {"LIST_APPEND", "SET_ADD", "MAP_ADD"}


@overload
def _instruction(name: Literal["LOAD_CONST", "KW_NAMES"], arg: object, *, source: Instr) -> Instr: ...


@overload
def _instruction(name: str, arg: InstrArg | None = None, *, source: Instr) -> Instr: ...


def _instruction(name: str, arg: object = None, *, source: Instr) -> Instr:
    if arg is None:
        return Instr(name, location=source.location)
    # bytecode 的 InstrArg 未包含任意常量；只有常量 opcode 的重载允许 object。
    return Instr(name, cast(InstrArg, arg), location=source.location)


def _site(code: CodeType, instruction: Instr, detail: str = "") -> Site:
    location = instruction.location
    line = (location.lineno if location else None) or code.co_firstlineno
    return (os.path.abspath(code.co_filename), code.co_qualname, line,
            (location.end_lineno if location else None) or line,
            (location.col_offset if location else None) or 0,
            (location.end_col_offset if location else None) or 0, detail)


def _emit_dispatch(dispatch: Dispatch, kind: str, site: Site, source: Instr,
                   *, existing_null: bool = False) -> list[Instr]:
    """消费栈顶 operands tuple，留下 dispatch 的返回值。"""
    @overload
    def ins(name: Literal["LOAD_CONST"], arg: object) -> Instr: ...
    @overload
    def ins(name: str, arg: InstrArg | None = None) -> Instr: ...
    def ins(name: str, arg: object = None) -> Instr:
        if name == "LOAD_CONST":
            return _instruction("LOAD_CONST", arg, source=source)
        return _instruction(name, cast(InstrArg | None, arg), source=source)
    result: list[Instr] = []
    if existing_null:
        # 输入 [NULL, operands]，整理为当前解释器所需的调用前缀。
        result.append(ins("LOAD_CONST", dispatch))
        if _NULL_FIRST:
            result.append(ins("SWAP", 2))
        else:
            result.extend((ins("SWAP", 3), ins("SWAP", 2)))
    else:
        if _NULL_FIRST:
            result.extend((ins("PUSH_NULL"), ins("LOAD_CONST", dispatch)))
        else:
            result.extend((ins("LOAD_CONST", dispatch), ins("PUSH_NULL")))
        # [operands, prefix...] -> [prefix..., operands]
        result.extend((ins("SWAP", 2), ins("SWAP", 3)))
    # [prefix, operands, kind, site] -> [prefix, kind, site, operands]
    result.extend((ins("LOAD_CONST", kind), ins("LOAD_CONST", site),
                   ins("SWAP", 2), ins("SWAP", 3)))
    result.extend(Instr(i.name, i.arg, location=source.location) for i in _CALL)
    return result


def _expanded(instructions: Bytecode) -> list[BytecodeEntry]:
    """拆开 3.13/3.14 的融合 fast-local 指令，复用单一存储探针实现。"""
    result: list[BytecodeEntry] = []
    pairs = {
        "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
        "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
        "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
        "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW"),
    }
    for instruction in instructions:
        if isinstance(instruction, Instr) and instruction.name in pairs:
            arguments = cast(tuple[str | CellVar | FreeVar, str | CellVar | FreeVar], instruction.arg)
            result.extend(Instr(name, arg, location=instruction.location)
                          for name, arg in zip(pairs[instruction.name], arguments))
        else:
            result.append(instruction)
    return result


class CodeTransformer:
    """每个 code 对象独立保存协议状态，生成的探针不依赖此转换器。"""

    def __init__(self, code: CodeType, dispatch: Dispatch, origin: OriginSpec,
                 observe_containers: bool = True) -> None:
        self.code = code
        self.dispatch = dispatch
        self.origin = origin
        self.observe_containers = observe_containers
        self.bytecode = Bytecode.from_code(code)
        self.output: list[BytecodeEntry] = []
        # 只追踪调用协议标记，不追踪对象；未知控制流入口清空标记，宁可漏报。
        self.stack: list[object] = []
        self.pending_keywords: tuple[str, ...] = ()
        self.keywords_index: int | None = None
        self.wants_call = getattr(getattr(dispatch, "__self__", None), "wants_call", None)
        # 内联推导式用 NULL 保存尚未绑定的局部槽位；NULL 不能传入 Python tuple。
        self.cleared_locals = {item.arg for item in self.bytecode
                               if isinstance(item, Instr) and item.name == "LOAD_FAST_AND_CLEAR"}

    def transform(self) -> CodeType:
        for instruction in _expanded(self.bytecode):
            if not isinstance(instruction, Instr):
                # Label 与异常区域注解原样保留，不将探针移出原来的保护范围。
                self.output.append(instruction)
                if isinstance(instruction, Label):
                    self.stack.clear()
                continue
            if instruction.name == "LOAD_CONST" and isinstance(instruction.arg, CodeType):
                instruction = _instruction("LOAD_CONST", transform(instruction.arg, self.dispatch, self.origin,
                                                                  self.observe_containers),
                                           source=instruction)
            if not self.observe_containers:
                # 不产生语义代理或 CALL 判定探针，而不是运行后丢弃事件。
                if instruction.name in _BINDING_NAMES:
                    self._transform_operation(instruction, [])
                else:
                    self.output.append(instruction)
                continue
            if instruction.name == "KW_NAMES":
                self.pending_keywords = cast(tuple[str, ...], instruction.arg)
                self.keywords_index = len(self.output)
                self.output.append(instruction)
                continue

            # 在改写前计算原指令的协议状态，探针自身不参与此抽象栈。
            before = self.stack.copy() if instruction.name in _CALL_NAMES else self.stack
            self._advance_protocol(instruction)
            self._transform_operation(instruction, before)
            if instruction.is_final() or instruction.has_jump():
                self.stack.clear()

        self.bytecode[:] = self.output
        return self.bytecode.to_code()

    def _report(self, instruction: Instr, reason: str) -> None:
        site = _site(self.code, instruction)
        unsupported.append((site[0], site[1], site[2], reason))

    def _observe(self, kind: str, site: Site, instruction: Instr) -> None:
        self.output.extend(_emit_dispatch(self.dispatch, kind, site, instruction))
        self.output.append(_instruction("POP_TOP", source=instruction))

    def _semantic(self, kind: str, count: int, site: Site, instruction: Instr) -> None:
        self.output.append(_instruction("BUILD_TUPLE", count, source=instruction))
        self.output.extend(_emit_dispatch(self.dispatch, kind, site, instruction))

    def _advance_protocol(self, instruction: Instr) -> None:
        name, argument = instruction.name, instruction.arg
        # Instr.arg 不随 opcode 自动收窄；以下 cast 对应 bytecode 校验过的操作数协议。
        stack = self.stack
        if name == "PUSH_NULL":
            stack.append(_NULL)
        elif name in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_DEREF", "LOAD_BUILD_CLASS"}:
            loaded_name = cast(tuple[bool, str], argument)[1] if isinstance(argument, tuple) else str(getattr(argument, "name", argument))
            value_marker = _FRAME if name == "LOAD_BUILD_CLASS" or loaded_name in _FRAME_NAMES else _VALUE
            flag = name == "LOAD_GLOBAL" and isinstance(argument, tuple) and cast(tuple[bool, str], argument)[0]
            if flag:
                stack.extend((_NULL, value_marker) if _NULL_FIRST else (value_marker, _NULL))
            else:
                stack.append(value_marker)
        elif name == "COPY":
            argument = cast(int, argument)
            stack.append(stack[-argument] if len(stack) >= argument else _VALUE)
        elif name == "SWAP":
            argument = cast(int, argument)
            if len(stack) >= argument:
                stack[-1], stack[-argument] = stack[-argument], stack[-1]
            else:
                stack.clear()
        elif name in {"LOAD_ATTR", "LOAD_METHOD"}:
            if stack:
                stack.pop()
            method = name == "LOAD_METHOD" or (isinstance(argument, tuple) and cast(tuple[bool, str], argument)[0])
            attribute_name = cast(tuple[bool, str], argument)[1] if isinstance(argument, tuple) else cast(str, argument)
            value_marker = _FRAME if attribute_name in _FRAME_NAMES else _VALUE
            stack.extend(((_NULL, value_marker) if _NULL_FIRST else (value_marker, _NULL)) if method else (value_marker,))
        else:
            consumed, produced = instruction.pre_and_post_stack_effect(jump=False)
            if consumed < 0:
                del stack[max(0, len(stack) + consumed):]
            elif consumed > 0:
                stack.extend([_VALUE] * consumed)
            if produced < 0:
                del stack[max(0, len(stack) + produced):]
            else:
                stack.extend([_VALUE] * produced)

    def _transform_operation(self, instruction: Instr, before: list[object]) -> None:
        name, argument = instruction.name, instruction.arg
        site = _site(self.code, instruction)
        if name in _BINDING_NAMES:
            self._binding(instruction)
        elif name in {"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR"}:
            self._attribute(instruction)
        elif name in _SUBSCRIPT_NAMES or (name == "BINARY_OP" and getattr(argument, "name", "") == "SUBSCR"):
            self._subscript(instruction, site)
        elif name in _BUILD_NAMES:
            self._build_container(instruction, site)
        elif name in _COMPREHENSION_NAMES:
            self._comprehension(instruction, site)
        elif name in _CALL_NAMES:
            self._call(instruction, site, before)
        else:
            self.output.append(instruction)
            if name in {"LOAD_SUPER_ATTR", "LIST_EXTEND", "SET_UPDATE", "DICT_UPDATE", "DICT_MERGE"}:
                self._report(instruction, f"{name}: provenance observation unavailable")

    def _binding(self, instruction: Instr) -> None:
        argument = instruction.arg
        if instruction.name == "STORE_FAST" and argument in self.cleared_locals:
            self.output.append(instruction)
            self._report(instruction, "comprehension local restoration: binding observation unavailable")
            return
        variable = str(getattr(argument, "name", argument))
        site = _site(self.code, instruction, variable)
        selected = (site[0] == os.path.abspath(self.origin[0])
                    and self.code.co_qualname == self.origin[1] and variable == self.origin[3]
                    and site[2] <= self.origin[2] <= site[3])
        # 先复制，成功 STORE 后才观察；避免重新读取 name/deref 改变语义。
        self.output.extend((_instruction("COPY", 1, source=instruction), instruction,
                            _instruction("BUILD_TUPLE", 1, source=instruction)))
        self._observe("capture" if selected else "bind", site, instruction)

    def _attribute(self, instruction: Instr) -> None:
        name, argument = instruction.name, instruction.arg
        if name == "STORE_ATTR":
            self._semantic("store_attribute", 2, _site(self.code, instruction, cast(str, argument)), instruction)
            self.output.append(_instruction("POP_TOP", source=instruction))
            return
        attribute = cast(tuple[bool, str], argument)[1] if isinstance(argument, tuple) else cast(str, argument)
        method = name == "LOAD_METHOD" or (isinstance(argument, tuple) and cast(tuple[bool, str], argument)[0])
        self._semantic("attribute", 1, _site(self.code, instruction, attribute), instruction)
        if method:
            self.output.append(_instruction("PUSH_NULL", source=instruction))
            if _NULL_FIRST:
                self.output.append(_instruction("SWAP", 2, source=instruction))

    def _subscript(self, instruction: Instr, site: Site) -> None:
        name = instruction.name
        storing = name in {"STORE_SUBSCR", "STORE_SLICE"}
        if name in {"BINARY_SLICE", "STORE_SLICE"}:
            self.output.append(_instruction("BUILD_SLICE", 2, source=instruction))
        self._semantic("store_subscript" if storing else "subscript", 3 if storing else 2, site, instruction)
        if storing:
            self.output.append(_instruction("POP_TOP", source=instruction))

    def _build_container(self, instruction: Instr, site: Site) -> None:
        name, argument = instruction.name, cast(int, instruction.arg)
        count = argument * 2 if name == "BUILD_MAP" else argument + 1 if name == "BUILD_CONST_KEY_MAP" else argument
        kind = "build_const_map" if name == "BUILD_CONST_KEY_MAP" else name.lower()
        self._semantic(kind, count, site, instruction)

    def _comprehension(self, instruction: Instr, site: Site) -> None:
        name, argument = instruction.name, cast(int, instruction.arg)
        # 暂存的观察 tuple 插在原操作数下方；原 opcode 成功后才发事件。
        mapping = name == "MAP_ADD"
        self.output.append(_instruction("COPY", argument + (2 if mapping else 1), source=instruction))
        if mapping:
            self.output.extend((_instruction("COPY", 3, source=instruction), _instruction("COPY", 3, source=instruction)))
        else:
            self.output.append(_instruction("COPY", 2, source=instruction))
        self.output.append(_instruction("BUILD_TUPLE", 3 if mapping else 2, source=instruction))
        if mapping:
            self.output.extend((_instruction("SWAP", 3, source=instruction), _instruction("SWAP", 2, source=instruction)))
        else:
            self.output.append(_instruction("SWAP", 2, source=instruction))
        self.output.append(_instruction(name, argument + 1, source=instruction))
        self._observe(name.lower(), site, instruction)

    def _call(self, instruction: Instr, site: Site, before: list[object]) -> None:
        name, argument = instruction.name, instruction.arg
        extra = 1 if name == "CALL_KW" else 0
        # 3.14 的 CALL_FUNCTION_EX 无标志参数，kwargs 槽位始终存在。
        ex_arity = -instruction.pre_and_post_stack_effect()[0] - 2 if name == "CALL_FUNCTION_EX" else 0
        arity = ex_arity if name == "CALL_FUNCTION_EX" else cast(int, argument) + extra
        null_depth = arity + (2 if _NULL_FIRST else 1)
        known_null = len(before) >= null_depth and before[-null_depth] is _NULL
        callable_depth = arity + (1 if _NULL_FIRST else 2)
        frame_sensitive = len(before) >= callable_depth and before[-callable_depth] is _FRAME
        if not known_null or frame_sensitive or self.wants_call is None:
            self.output.append(instruction)
            if not known_null:
                self._report(instruction, f"{name}: implicit or unknown callable protocol")
        else:
            self._conditional_call(instruction, site, before, callable_depth, ex_arity)
        self.pending_keywords = ()
        self.keywords_index = None

    def _conditional_call(self, instruction: Instr, site: Site, before: list[object],
                          callable_depth: int, ex_arity: int) -> None:
        native, complete = Label(), Label()
        if self.keywords_index is not None:
            del self.output[self.keywords_index]
        self._call_guard(instruction, callable_depth, native)
        if instruction.name == "CALL_FUNCTION_EX":
            self._pack_extended_call(instruction, before, ex_arity)
            kind = "call_ex"
        else:
            self._pack_call(instruction)
            kind = "call"
        self.output.extend(_emit_dispatch(self.dispatch, kind, site, instruction, existing_null=True))
        self.output.append(_instruction("JUMP_FORWARD", complete, source=instruction))
        self.output.append(native)
        if self.keywords_index is not None:
            self.output.append(_instruction("KW_NAMES", self.pending_keywords, source=instruction))
        self.output.append(instruction)
        self.output.append(complete)

    def _call_guard(self, instruction: Instr, callable_depth: int, native: Label) -> None:
        # 原参数和 NULL 留在栈上；只复制 callable 给无副作用的身份判定。
        self.output.append(_instruction("COPY", callable_depth, source=instruction))
        if _NULL_FIRST:
            self.output.extend((_instruction("PUSH_NULL", source=instruction),
                                _instruction("LOAD_CONST", self.wants_call, source=instruction)))
        else:
            self.output.extend((_instruction("LOAD_CONST", self.wants_call, source=instruction),
                                _instruction("PUSH_NULL", source=instruction)))
        self.output.extend((_instruction("SWAP", 2, source=instruction), _instruction("SWAP", 3, source=instruction)))
        self.output.extend(Instr(i.name, 1, location=instruction.location) for i in _CALL)
        jump = "POP_JUMP_FORWARD_IF_FALSE" if "POP_JUMP_FORWARD_IF_FALSE" in __import__("opcode").opmap else "POP_JUMP_IF_FALSE"
        self.output.append(_instruction(jump, native, source=instruction))

    def _pack_extended_call(self, instruction: Instr, before: list[object], ex_arity: int) -> None:
        if ex_arity == 2 and before and before[-1] is _NULL:
            # 3.14 空 kwargs 是解释器 NULL，不是可放入 tuple 的 PyObject。
            self.output.extend((_instruction("POP_TOP", source=instruction),
                                _instruction("BUILD_MAP", 0, source=instruction)))
        if ex_arity == 1:
            self.output.append(_instruction("BUILD_MAP", 0, source=instruction))
        if not _NULL_FIRST:
            # [F,NULL,args,kwargs] -> [NULL,F,args,kwargs]
            self._move_null_below_callable(instruction)
        self.output.append(_instruction("BUILD_TUPLE", 3, source=instruction))

    def _pack_call(self, instruction: Instr) -> None:
        argument = cast(int, instruction.arg)
        if instruction.name == "CALL_KW":
            # 将名字 tuple 移到参数下面，再一次性打包参数。
            self.output.extend(_instruction("SWAP", depth, source=instruction)
                               for depth in range(argument + 1, 1, -1))
        self.output.append(_instruction("BUILD_TUPLE", argument, source=instruction))
        if instruction.name == "CALL_KW":
            self.output.append(_instruction("SWAP", 2, source=instruction))
        else:
            self.output.append(_instruction("LOAD_CONST", self.pending_keywords, source=instruction))
        if not _NULL_FIRST:
            # [F,NULL,args,keys] -> [NULL,F,args,keys]
            self._move_null_below_callable(instruction)
        self.output.append(_instruction("BUILD_TUPLE", 3, source=instruction))
        # KW_NAMES 已移到保留原 CALL 的分支，避免影响探针调用。

    def _move_null_below_callable(self, instruction: Instr) -> None:
        self.output.extend((_instruction("SWAP", 3, source=instruction),
                            _instruction("SWAP", 4, source=instruction),
                            _instruction("SWAP", 3, source=instruction)))


def transform(code: CodeType, dispatch: Dispatch, origin: OriginSpec,
              observe_containers: bool = True) -> CodeType:
    """递归转换 code 常量；关闭载体观察时只插入绑定／捕获探针。"""
    return CodeTransformer(code, dispatch, origin, observe_containers).transform()

let s:repo_dir = fnamemodify(finddir('ai-infra-dev', getcwd() . ';'), ':p')
let g:noterepo_dir = s:repo_dir . 'note/'
let g:coderepo_dir = [['vllm/vllm', s:repo_dir . 'code/vllm/']]

# Language-model-with-Attention
seq2word+Attention

训练
训练的代码在seq2word_split_model下

执行： 

nohup ./start_train.sh ${data_vocab_path} ${model_save_path} ${graph_save_path} ${config_file} > train.log &

参数解释：
${data_vocab_path} 上述生成的训练数据目录
${model_save_path} 模型参数保存路径
${graph_save_path} 模型保存路径，最后会保存三种模型，即后缀分别为lm、kc_full和kc_slim的模型，实际上只需要用到kc_slim模型
${config_file} 参数配置文件

测试
执行：

nohup python test.py ${graph_file} ${data_vocab_path} ${full_vocab_file} ${config_file} ${test_file} > test.log &

参数解释：
${graph_file} 模型文件，即保存的kc_slim模型  
${data_vocab_path} 词表路径
${full_vocab_file} 16万大词表文件
${config_file} 参数配置文件
${test_file} 测试文件

测试结果将输出在"test_result"文件中。

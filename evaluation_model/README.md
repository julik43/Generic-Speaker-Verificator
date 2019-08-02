# Evaluation of models

To evaluate the models you need too:

1. Use the fill_BD.py code to create a database as known speakers for the evaluation, configuring the parameter of NFFT and SIZE_FFT as the model you are going to evaluate.

2. Configure the parameters exactly as the model you trained with

3. Update the path of the saver with the correct path to the final weights of your model.

e.g. saver.restore(sess, "module_vgg.py_0.01_10/final_weights.ckpt")

4. Run the model with the desired database and files:

python heatmap_module_vgg.py --verification_file verification_file.txt ----unk_file file_unknown.txt --database data_bases/database_32_librireal
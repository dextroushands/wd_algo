import tensorflow as tf
from tensorflow.python.framework import ops
from deep_regression.model import RegressionModel
import os
from deep_regression.utils import mean, get_metrics
from deep_regression.data_helper import TrainData
import json

class Trainer(object):

    def __init__(self, config):

        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None

        self.n_layers = len(config['hidden_size'])
        self.config = config
        self.load_data()
        self.train_inputs, self.eval_inputs, self.train_labels, self.eval_labels, self.input_size, _, _= self.train_data_obj.data_pre()
        print("train data size: {}".format(len(self.train_labels)))
        # self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))
        print("input_size: {}".format(self.input_size))

        self.output_size = 1
        self.build_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        self.train_data_obj = TrainData(self.config)

        # 生成验证集对象和验证集数据
        # self.eval_data_obj = EvalData(self.config)

    def build_model(self):
        self.model = RegressionModel(self.config, self.input_size)

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                              self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                             self.config["output_path"] + "/summary/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            # eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path)


            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.train_data_obj.next_batch(self.train_inputs, self.train_labels,
                                                            self.config["batch_size"]):
                    summary, loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])
                    # tf.summary.scalar("loss0", loss)
                    # summary_op1 = tf.summary.merge_all()
                    # train_summary_writer.add_summary(summary_op1)
                    train_summary_writer.add_summary(summary, current_step)

                    var_score, mse, mae = get_metrics(y_pred=predictions, y_true=batch["y"])
                    print(
                        "train: step: {}, loss: {}, var_score: {}, mse: {}, mae: {}".format(
                            current_step, loss, var_score, mse, mae))


                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.train_data_obj.next_batch(self.eval_inputs, self.eval_labels,
                                                                        self.config["batch_size"]):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary, current_step)

                            eval_losses.append(eval_loss)

                            var_score, mse, mae = get_metrics(y_pred=eval_predictions,y_true=eval_batch["y"])
                            eval_accs.append(var_score)
                            eval_aucs.append(mse)
                            eval_recalls.append(mae)
                            # eval_precs.append(msle)

                        print("\n")
                        print("eval:  loss: {}, var_score: {}, mse: {}, mae: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

if __name__=='__main__':
    config_path = 'deep_regression/config.json'
    with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config_path), "r") as fr:
        config = json.load(fr)
    trainer = Trainer(config)
    trainer.train()


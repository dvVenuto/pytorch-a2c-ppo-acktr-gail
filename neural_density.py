from GatedPixelCNN.func import process_density_images, process_density_input, get_network
import tensorflow as tf


import pickle


#frame = imresize(screen / self.img_scale, (42, 42), order=1)
#image scale = 255.

class NeuralDensity:
    def __init__(self,sess):
        self.sess=sess
        self.density_model = get_network("density")

    def neural_psc(self, frame, step):
        last_frame = process_density_images(frame)
        density_input = process_density_input(last_frame)

        prob = self.density_model.prob_evaluate(self.sess, density_input, True) + 1e-8
        prob_dot = self.density_model.prob_evaluate(self.sess, density_input) + 1e-8
        pred_gain = np.sum(np.log(prob_dot) - np.log(prob))
        psc_reward = pow((exp(0.1 * pow(step + 1, -0.5) * max(0, pred_gain)) - 1), 0.5)
        return psc_reward

    def saveModel(self,path):
        f = open('neuralDensityModel' + str(path) +'.pkl', 'wb')
        pickle.dump(self.density_model, f)

def run(queue, name, epochs=100, verbose=1):
    from Config import MyConfig,keras,tf
    from Model import MyModel

    config = MyConfig(name=name, epochs=epochs, verbose=verbose, clean=False)
    print tf.get_default_graph()
    # device = 0
    # with tf.device(device):
    model = MyModel(config=config, model=keras.models.load_model(config.model_path))
    print "start train {}".format(name)
    with config.sess.as_default():
        score = model.comp_fit_eval()
    queue.put((name, score))
    keras.models.save_model(config.model_path, model)



if __name__ == '__main__':
    import multiprocessing as mp

    queue = mp.Queue()
    run(queue, name='ga')

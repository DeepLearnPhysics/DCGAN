import tensorflow as tf
import sys, os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from classifier import build_classifier

###################################
# Build a classifier for mnist data
###################################

# Prepare and input variable:

data_tensor = tf.placeholder(tf.float32, [None, 784], name='x')
label_tensor = tf.placeholder(tf.float32, [None, 10], name='labels')
x = tf.reshape(data_tensor, (tf.shape(data_tensor)[0], 28, 28, 1))


# Build a residual network on top of it, nothing complicated:
logits = build_classifier(input_tensor=x, 
                          n_output_classes=10,
                          is_training=True)

###################################
# Set up training parameters:     #
###################################
BASE_LEARNING_RATE = 0.001
LOGDIR='/home/cadams/mnist_gan/logs'
RESTORE=False
SAVE_ITERATION=500

# Add a global step accounting for saving and restoring training:
with tf.name_scope("global_step") as scope:
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='global_step')

# Add cross entropy (loss)
with tf.name_scope("cross_entropy") as scope:
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,
                                                logits=logits))
    loss_summary = tf.summary.scalar("Loss", cross_entropy)

# Add accuracy:
with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.summary.scalar("Accuracy", accuracy)


# Set up a training algorithm:
with tf.name_scope("training") as scope:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(BASE_LEARNING_RATE).minimize(
            cross_entropy, global_step=global_step)




merged_summary = tf.summary.merge_all()

# Set up a supervisor to manage checkpoints and summaries

# sv = tf.train.Supervisor(logdir=LOGDIR, summary_op=None)

# # Set up a saver:
train_writer = tf.summary.FileWriter(LOGDIR)



print "Initialize session ..."
with tf.Session() as sess:
    
    if not RESTORE:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        saver = tf.train.Saver()
    else: 
        latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
        print latest_checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, latest_checkpoint)
        # 
        # exit()
        # 
        # saver = tf.train.import_meta_graph(LOGDIR+"/checkpoints/save-15")
        # checkpoints = saver.recover_last_checkpoints(LOGDIR+"/checkpoints/save-15")
        # print checkpoints

        # saver.restore(sess,tf.train.latest_checkpoint('./'))

# with sv.managed_session() as sess:

    print "Begin training ..."
    # Run training loop
    # while not sv.should_stop():
    for i in xrange(500):
        step = sess.run(global_step)

        # Receive data (this will hang if IO thread is still running = this
        # will wait for thread to finish & receive data)
        data, label = mnist.train.next_batch(32)


        if step != 0 and step % SAVE_ITERATION == 0:
            print saver.save(
                sess,
                LOGDIR+"/checkpoints/save",
                global_step=step)

        # print training accuracy every 10 steps:
        # if i % 10 == 0:
        #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary],
        #                                                       feed_dict={data_tensor:data,
        #                                                                  label_tensor:label})
        #     train_writer.add_summary(loss_s,i)
        #     train_writer.add_summary(accuracy_s,i)

            # sys.stdout.write('Training in progress @ step %d accuracy %g\n' % (i,training_accuracy))
            # sys.stdout.flush()

        # if step != 0 and step % 5 == 0:
        #     print "Running Summary"
        #     _, summ = sess.run([merged_summary])
        #     print "Saving Summary"
        #     sv.summary_computed(sess, summ)
        [loss, acc, summ, _] = sess.run([cross_entropy, accuracy, merged_summary, train_step], 
                                  feed_dict={data_tensor: data, 
                                             label_tensor: label})
        train_writer.add_summary(summ, step)



        # train_writer.add_summary(summary, i)
        sys.stdout.write(
            'Training in progress @ step %d, loss %g, accuracy %g\n' % (step, loss, acc))
        # sys.stdout.flush()
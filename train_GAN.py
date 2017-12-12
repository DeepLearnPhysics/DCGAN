import tensorflow as tf
import sys, os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from discriminator import build_discriminator
from generator import build_generator
import numpy

###################################
# Set up training parameters:     #
###################################
BASE_LEARNING_RATE = 0.00001
LOGDIR='/home/cadams/DCGAN/logs/GAN'
RESTORE=False
SAVE_ITERATION=500
TRAIN_STEPS=100000
BATCH_SIZE=128

###################################
# Build a classifier for mnist data
###################################

# Prepare and input variable:

# Input noise to the generator:
noise_tensor = tf.placeholder(tf.float32, [int(BATCH_SIZE*0.5), 10*10], name="noise")
gen_input    = tf.reshape(noise_tensor, (tf.shape(noise_tensor)[0], 10,10, 1))

# Placeholder for the discriminator input:
data_tensor  = tf.placeholder(tf.float32, [int(BATCH_SIZE*0.5), 784], name='x')
# label_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='labels')
x            = tf.reshape(data_tensor, (tf.shape(data_tensor)[0], 28, 28, 1))


# Build a residual network on top of it, nothing complicated:
# "Real" data logits:
real_logits = build_discriminator(input_tensor=x, reuse=False)

gen_images = build_generator(input_tensor=gen_input)
gen_images = tf.reshape(gen_images, (tf.shape(gen_images)[0], 28, 28, 1))


# "Fake" data logits:
fake_logits = build_discriminator(input_tensor=gen_images, reuse=True)


# Add a global step accounting for saving and restoring training:
with tf.name_scope("global_step") as scope:
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='global_step')



# Build the loss functions:
# Add cross entropy (loss)
with tf.name_scope("cross_entropy") as scope:

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
        labels = tf.ones_like(real_logits)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
        labels = tf.zeros_like(fake_logits)))

    d_loss_total = d_loss_real + d_loss_fake

    # This is the adverserial step: g_loss tries to optimize fake_logits to one,
    # While d_loss_fake tries to optimize fake_logits to zero.
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
        labels = tf.ones_like(fake_logits)))
    d_loss_summary = tf.summary.scalar("Discriminator Real Loss", d_loss_real)
    d_loss_summary = tf.summary.scalar("Discriminator Fake Loss", d_loss_fake)
    d_loss_summary = tf.summary.scalar("Discriminator Total Loss", d_loss_total)
    d_loss_summary = tf.summary.scalar("Generator Loss", g_loss)





# Add accuracy:
with tf.name_scope("accuracy") as scope:
    # Compute the discriminator accuracy on real data, fake data, and total:
    accuracy_real  = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_logits), 
                                                     tf.ones_like(real_logits)), 
                                            tf.float32))
    accuracy_fake  = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_logits), 
                                                     tf.zeros_like(fake_logits)), 
                                            tf.float32))
    total_accuracy = 0.5*(accuracy_fake +  accuracy_real)
    
    acc_real_summary = tf.summary.scalar("Real Accuracy", accuracy_real)
    acc_real_summary = tf.summary.scalar("Fake Accuracy", accuracy_fake)
    acc_real_summary = tf.summary.scalar("Total Accuracy", total_accuracy)

#Save some images:
snap_shot_real = tf.reshape(x, (-1, 28, 28, 1))
snap_shot_fake = tf.reshape(gen_images, (-1, 28, 28, 1))
tf.summary.image('fake_images', snap_shot_fake, max_outputs=4)
tf.summary.image('real_images', snap_shot_real, max_outputs=4)


# Set up a training algorithm:
with tf.name_scope("training") as scope:
    # Make sure the optimizers are only operating on their own variables:
    all_variables = tf.trainable_variables()
    discriminator_vars = [v for v in all_variables if v.name.startswith('discriminator/')]
    generator_vars     = [v for v in all_variables if v.name.startswith('generator/')]
    
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        discriminator_optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE, 0.5).minimize(
            d_loss_total, global_step=global_step, var_list=discriminator_vars)
        generator_optimizer     = tf.train.AdamOptimizer(BASE_LEARNING_RATE, 0.5).minimize(
            g_loss, global_step=global_step, var_list=generator_vars)


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


    print "Begin training ..."
    # Run training loop
    # while not sv.should_stop():
    for i in xrange(TRAIN_STEPS):
        step = sess.run(global_step)

        # Receive data (this will hang if IO thread is still running = this
        # will wait for thread to finish & receive data)
        
        # Prepare the input to the networks:
        real_data, label = mnist.train.next_batch(int(BATCH_SIZE*0.5))

        real_data = 2*(real_data - 0.5)        


        fake = numpy.random.uniform(-1, 1, (int(BATCH_SIZE*0.5), 10*10))


        # Update the generator:
        [ summary, g_l, d_l_r, acc, _ ] = sess.run([merged_summary, 
                                              g_loss,
                                              d_loss_fake, 
                                              total_accuracy, 
                                              generator_optimizer], 
                         feed_dict = {noise_tensor: fake,
                                      data_tensor : real_data})
        # if the discriminator accuracy is below 50%, update the discriminator:

        # Update the discriminator:
        fake = numpy.random.uniform(-1, 1, (int(BATCH_SIZE*0.5), 10*10))
        [generated_mnist, _] = sess.run([gen_images, 
                                        discriminator_optimizer], 
                                        feed_dict = {noise_tensor : fake,
                                                     data_tensor : real_data})






        train_writer.add_summary(summary, step)


        if step != 0 and step % SAVE_ITERATION == 0:
            saver.save(
                sess,
                LOGDIR+"/checkpoints/save",
                global_step=step)


        # train_writer.add_summary(summary, i)
        # sys.stdout.write('Training in progress @ step %d\n' % (step))
        if step % 50 == 0:
            print 'Training in progress @ step %d, g_loss %g, d_loss %g accuracy %g \n' % (step, g_l, d_l_r, acc)

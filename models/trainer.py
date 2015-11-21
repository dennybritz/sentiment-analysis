import tensorflow as tf
import time
import datetime
import os


class Trainer:
    def __init__(self, train_op, global_step, summary_op, eval_feed_dict,
                 save_every=10, evaluate_every=10):
        self.train_op = train_op
        self.global_step = global_step
        self.summary_op = summary_op
        self.save_every = save_every
        self.evaluate_every = evaluate_every
        self.eval_feed_dict = eval_feed_dict

        # All data goes in here
        RUNDIR = "./runs/%s" % int(time.time())

        # Write graph
        # gsd = GraphSerDe()
        # gsd.serialize("%s/graph" % RUNDIR)

        # Initialize summary writers
        sess = tf.get_default_session()
        self.train_writer = tf.train.SummaryWriter(
            "%s/summaries/train" % RUNDIR, sess.graph_def)
        self.eval_writer = tf.train.SummaryWriter(
            "%s/summaries/eval" % RUNDIR, sess.graph_def)

        # Initialize saver
        self.save_prefix = "%s/checkpoints/model" % RUNDIR
        if not os.path.exists(os.path.dirname(self.save_prefix)):
            os.makedirs(os.path.dirname(self.save_prefix))
        self.saver = tf.train.Saver(tf.all_variables())

    def evaluate(self):
        sess = tf.get_default_session()
        summaries_, global_step_ = sess.run([self.summary_op, self.global_step], feed_dict=self.eval_feed_dict)
        self.eval_writer.add_summary(summaries_, global_step_)
        # Print summaries
        print("\n========== Evaluation ==========")
        summary_obj = tf.Summary.FromString(summaries_)
        interesting_summaries = [v for v in summary_obj.value if "queue/" not in v.tag]
        print "\n".join(["{}: {:f}".format(v.tag, v.simple_value) for v in interesting_summaries])
        print("")

    def step(self):
        sess = tf.get_default_session()
        # Run training step
        _, global_step_, summaries_ = sess.run([self.train_op, self.global_step, self.summary_op])
        print("{}: Step {}".format(datetime.datetime.now().isoformat(), global_step_))
        # Write summary
        self.train_writer.add_summary(summaries_, global_step_)
        # Maybe save
        if global_step_ % self.save_every == 0:
            save_path = self.saver.save(sess, self.save_prefix, global_step_)
            print("\nSaved model parameters to %s" % save_path)
        # Maybe evaluate
        if global_step_ % self.evaluate_every == 0:
            self.evaluate()

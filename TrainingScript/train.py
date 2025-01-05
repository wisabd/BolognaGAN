from model.chi2calc import Chi2Metric
from common.VoxInputParameters import VoxInputParameters
from common.XMLHandler import XMLHandler
from common.GANInputParameters import GANInputParametersFromXML
from common.DataParameters import DataParametersFromXML
from common.TrainingInputParameters import TrainingInputParameters
from common.DataReader import DataLoader
import slurm
from tensorflow.python.distribute.distribute_lib import Strategy

import numpy as np
import ROOT

logger = logging.getLogger("training")

PROFILE = os.getenv("PROFILE", "FALSE") == "TRUE"


def select_distributed_strategy(no_slurm: bool, force_gpu: bool) -> Strategy:
    """Selects the distributed strategy to use based on the execution environment."""

    gpus = tf.config.list_physical_devices("GPU")
    has_gpu = len(gpus) > 0

    if force_gpu and not has_gpu:
        logger.error("No GPU available, but --force-gpu was specified. Exiting.")
        exit(1)

    if not no_slurm and slurm.running_on_slurm():

        slurm_ports = slurm.get_var("STEP_RESV_PORTS")
        assert slurm_ports is not None, "No SLURM ports allocated"

        slurm_ports = slurm_ports.split("-")
        slurm_ports = [int(port) for port in slurm_ports]
        assert len(slurm_ports) > 0, "No ports allocated by SLURM"

        if has_gpu:
            logger.info("Running on SLURM with GPU, using MultiWorkerMirroredStrategy")
            resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=slurm_ports[0])

        else:
            logger.info("Running on SLURM without GPU, using custom MultiWorkerMirroredStrategy")
            resolver = slurm.SlurmClusterCPUResolver(port_base=slurm_ports[0])

        logger.debug("Cluster configuration: %s", resolver.cluster_spec())

        return tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=resolver)

    elif len(gpus) > 1:
        logger.debug("GPUs available, using MirroredStrategy")
        return tf.distribute.MirroredStrategy()

    else:
        logger.debug("Using default strategy")
        return tf.distribute.get_strategy()


def get_logger_format(no_slurm: bool = False):
    logging_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    if not no_slurm and slurm.running_on_slurm():
        job_id = slurm.get_job_id()
        local_id = slurm.get_local_id()
        node_name = platform.node()

        slurm_str = f"{job_id}@{node_name}[{local_id}]"

        logging_format = f"{slurm_str} {logging_format}"

    return logging_format


def max_rss():
    return getrusage(RUSAGE_SELF).ru_maxrss / 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gan")
    parser.add_argument("-ip", "--particle", default="")
    parser.add_argument("-e", "--firstEpoch", default="", type=int)
    parser.add_argument("-emin", "--eta_min", default="", type=int)
    parser.add_argument("-emax", "--eta_max", default="", type=int)
    parser.add_argument("-i", "--input_dir", default="")
    parser.add_argument("-r", "--energy_range", default="")
    parser.add_argument("-odg", "--output_dir_gan", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--binning", default="binning.xml")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("-si", "--sample-interval", type=int, default=1)
    parser.add_argument("--no-slurm", action="store_true")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--steps-per-execution", type=int, default=1)
    args = parser.parse_args()

    logging_format = get_logger_format(no_slurm=args.no_slurm)
    logging.basicConfig(level=logging.DEBUG, format=logging_format, datefmt="%Y-%m-%d %H:%M:%S")

    import tensorflow as tf
    import keras

    if slurm.running_on_slurm():
        keras.utils.disable_interactive_logging()

    logger.info("Running on tensorflow %s", tf.version.VERSION)

    # Define the strategy to distribute the model
    distr_strategy = select_distributed_strategy(args.no_slurm, args.force_gpu)

    particle = args.particle
    eta_min = args.eta_min
    eta_max = args.eta_max
    start_epoch = args.firstEpoch
    input_dir = args.input_dir
    output_dir_gan = args.output_dir_gan
    energy_range = args.energy_range
    xml_file_name = args.binning
    steps_per_execution = args.steps_per_execution

    epochs = args.epochs
    sample_interval = args.sample_interval
    training_strategy = "All"

    logger.debug("Loading data...")

    input_params = VoxInputParameters(
        input_dir,
        particle,
        eta_min,
        eta_max,
        xmlFileName=xml_file_name,
    )
    xml = XMLHandler(input_params)
    data_params = DataParametersFromXML(xml, energy_range)
    gan_params = GANInputParametersFromXML(xml, energy_range)
    training_params = TrainingInputParameters(start_epoch, epochs, sample_interval, output_dir_gan, training_strategy)

    data_loader = DataLoader(input_params, data_params)

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError  # type: ignore - Suppress ROOT warnings

    # if training_params.loadFromBaseline:
    #     logger.info("Loading from baseline")
    #     try:
    #         logger.info("Try to load checkpoint from baseline folder %s", training_params.loading_dir)
    #         wgan.saver.restore(
    #             os.path.join(
    #                 training_params.loading_dir,
    #                 f"model_{input_params.particle}_region_{input_params.region}",
    #             )
    #         )
    #         # self.saver.save_counter=1
    #     except:
    #         logger.info(
    #             "Error while loading baseline checkpoint at %s",
    #             training_params.loading_dir,
    #         )
    #         raise
    # else:
    #     logger.info("Training from scratch")

    # # Checkpoint directory
    # checkpoint_dir = os.path.join(
    #     training_params.GAN_dir,
    #     input_params.particle,
    #     f"checkpoints_eta_{input_params.eta_min}_{input_params.eta_max}",
    # )

    # tf.io.gfile.makedirs(checkpoint_dir)

    # if training_params.start_epoch > 0:
    #     try:
    #         logger.info("Try to load starting model %d" % (training_params.start_epoch))
    #         iepoch = str(int((training_params.start_epoch - 100000) / 2000 + 1))
    #         logger.info(
    #             "convert trainingInputs.start_epoch ",
    #             training_params.start_epoch,
    #             iepoch,
    #         )
    #         wgan.saver.restore(os.path.join(checkpoint_dir, f"model-{iepoch}"))
    #         training_params.start_epoch = training_params.start_epoch + 1

    #     except:
    #         logger.info("Error while loading starting model")
    #         raise

    # ind_of_exp = 1

    if training_params.training_strategy == "All":
        logger.info("Starting training for all samples")
        exp_max = data_params.max_expE
        exp_min = data_params.min_expE

    elif training_params.training_strategy == "Sequential":
        #     logger.info("Starting sequential training")
        #     logger.info("first sample trained for %d epochs", training_params.epochsForFirstSample)
        #     logger.info(
        #         "additional samples added at intervals of %d epochs",
        #         training_params.epochsForAddingASample,
        #     )
        #     exp_max = data_params.exp_mid
        #     exp_min = data_params.exp_mid
        raise NotImplementedError("Sequential training is not implemented")

    # for epoch in range(0, training_params.start_epoch):
    #     # after 50000, each 20000 it makes larger the reange
    #     if (
    #         training_params.training_strategy == "Sequential"
    #         and epoch >= training_params.epochsForFirstSample
    #         and (epoch - training_params.epochsForFirstSample) % training_params.epochsForAddingASample == 0
    #     ):
    #         if ind_of_exp == 0:
    #             exp_max += 1
    #             ind_of_exp += 1
    #         elif ind_of_exp > 0:
    #             if exp_max < data_params.max_expE:
    #                 exp_max += 1
    #                 ind_of_exp = -ind_of_exp
    #         else:
    #             if exp_min > data_params.min_expE:
    #                 exp_min -= 1
    #                 ind_of_exp = -ind_of_exp

    # TODO: sequential training is not implemented
    # we should do something like
    # if training_params.training_strategy == "Sequential":
    #     while epoch < training_params.max_epochs:
    #         load_dataset(exp_min, exp_max)
    #         model.train(tot_epochs, X, Labels)
    #         exp_max = min(exp_max + 1, data_params.max_expE)
    #         exp_min = max(exp_min - 1, data_params.min_expE)
    #         epoch += tot_epochs

    # # load data on first epoch
    # change_data = (epoch == train_params.start_epoch)

    # if (train_params.training_strategy == "Sequential"
    #     and epoch >= train_params.epochsForFirstSample
    #     and (epoch-train_params.epochsForFirstSample) % train_params.epochsForAddingASample == 0):

    #   #after 50000, each 20000 it makes larger the reange

    #   if ind_of_exp == 0 :
    #     self.exp_max += 1
    #     change_data = True
    #     ind_of_exp += 1
    #   elif ind_of_exp > 0:
    #     if self.exp_max <data_params.max_expE:
    #       self.exp_max += 1
    #       change_data = True
    #       ind_of_exp = -ind_of_exp
    #   else :
    #     if self.exp_min >data_params.min_expE:
    #       self.exp_min -= 1
    #       change_data = True
    #       ind_of_exp = -ind_of_exp

    logger.info("Loading train data (exp_min=%d, exp_max=%d)...", exp_min, exp_max)

    dur = time.perf_counter()
    samples, labels = data_loader.getAllTrainData(exp_min, exp_max)
    dur = time.perf_counter() - dur

    logger.info("Data loaded in %f seconds (RSS: %d MB)", dur, max_rss())

    logger.info("Starting epoch: %d", training_params.start_epoch)

    from model import bologan

    with distr_strategy.scope():

        discriminator = bologan.get_discriminator_model(
            hidden_layers=gan_params.discriminatorLayers,
            sample_dim=gan_params.nvoxels,
            labels_dim=gan_params.conditional_dim,
        )

        discriminator.summary()

        generator = bologan.get_generator_model(
            hidden_layers=gan_params.generatorLayers,
            noise_dim=gan_params.latent_dim,
            labels_dim=gan_params.conditional_dim,
            output_dim=gan_params.nvoxels,
            activation=gan_params.activationFunction,
            batch_normalization=gan_params.useBatchNormalisation,
        )

        generator.summary()

        wgan = bologan.WGANGP(
            generator=generator,
            discriminator=discriminator,
            latent_dim=gan_params.latent_dim,
            discriminator_steps=gan_params.n_disc,
            gp_weight=gan_params.lam,
        )

        generator_optimizer = keras.optimizers.Adam(learning_rate=gan_params.G_lr, beta_1=gan_params.G_beta1)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=gan_params.D_lr, beta_1=gan_params.D_beta1)

        wgan.compile(
            g_optimizer=generator_optimizer,
            d_optimizer=discriminator_optimizer,
            g_loss_fn=bologan.generator_loss,
            d_loss_fn=bologan.discriminator_loss,
            steps_per_execution=steps_per_execution,
        )

    logger.info("Converting data to numpy arrays...")
    dur = time.perf_counter()
    labels = np.array(labels, dtype=np.float32)
    samples = np.array(samples, dtype=np.float32)
    dur = time.perf_counter() - dur
    logger.info("Data converted to numpy arrays in %.2f seconds (RSS: %d MB)", dur, max_rss())

    logger.info("Running Model.fit...")

    tensorboard_dir = os.path.join(output_dir_gan, "logs")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        profile_batch=(10, 20) if PROFILE else 0,
        write_images=True,
    )

    checkpoints_path = os.path.join(output_dir_gan, "checkpoints")
    if start_epoch != 0:
        import glob

        modelfile = glob.glob(checkpoints_path + "/model-{start_epoch:04d}*")

        if len(modelfile) > 0:
            logger.info("Resuming training from epoch %d", start_epoch)

            logger.debug("Loading model from %s", modelfile[0])
            wgan.load_weights(modelfile[0])
        else:
            raise ValueError("No model found at %s. Cannot resume training. Maybe --firstEpoch is wrong?" % modelfile)

    best_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir_gan, "checkpoints", "model-best"),
        save_best_only=True,
        monitor="chi2_per_ndf",
        mode="min",
        save_weights_only=True,
        verbose=1,
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir_gan, "checkpoints", "model-{epoch:04d}-chi{chi2_per_ndf:.2f}"),
        save_weights_only=True,
    )

    backup_path = os.path.join(output_dir_gan, "backup")
    backup_callback = keras.callbacks.BackupAndRestore(backup_path)

    chi2_callback = Chi2Metric(
        vox_params=input_params,
        data_params=data_params,
        data_loader=data_loader,
        gan_params=gan_params,
        log_dir=tensorboard_dir,
    )

    callbacks = [
        chi2_callback,
        tensorboard_callback,
        backup_callback,
        best_checkpoint_callback,
        checkpoint_callback,
    ]

    wgan.fit(
        x=[samples, labels],
        batch_size=gan_params.batchsize,
        epochs=training_params.max_epochs,
        initial_epoch=training_params.start_epoch,
        callbacks=callbacks,
    )

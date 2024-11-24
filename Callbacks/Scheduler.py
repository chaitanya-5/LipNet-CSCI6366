def scheduler(epoch, lr):
    """
    Scheduler function to adjust learning rate based on epoch.

    Args:
        epoch (int): The current epoch number.
        lr (float): The current learning rate.

    Returns:
        float: The updated learning rate.
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

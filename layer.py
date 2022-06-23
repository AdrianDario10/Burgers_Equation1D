import tensorflow as tf


class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Burgers' equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, tx):
        """
        Computing 1st and 2nd derivatives for Burgers' equation.
        Args:
            x: input variable.
        Returns:
            model output, 1st and 2nd derivatives.
        """
        t, x = [tx[..., i, tf.newaxis] for i in range(tx.shape[-1])]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(t)
            ggg.watch(x)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(t)
                gg.watch(x)
                with tf.GradientTape(persistent=True) as g:
                    g.watch(t)
                    g.watch(x)
                    u = self.model(tf.concat([t, x], axis=-1))
                u_t = g.batch_jacobian(u, t)[..., 0]
                u_x = g.batch_jacobian(u, x)[..., 0]
                del g
            u_xx = gg.batch_jacobian(u_x, x)[..., 0]
            del gg
        u_xxx = ggg.batch_jacobian(u_xx, x)[..., 0]

        del ggg

        return u, u_t, u_x, u_xx

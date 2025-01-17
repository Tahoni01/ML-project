import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Dataset e Preprocessing
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(jnp.float32) / 255.0  # Normalizzazione [0, 1]
    x_test = x_test.astype(jnp.float32) / 255.0
    x_train = x_train.reshape(-1, 28 * 28)  # Flatten 28x28 immagini
    x_test = x_test.reshape(-1, 28 * 28)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_and_preprocess_data()

# 2. Definizione del Modello
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)  # Strato denso con 128 neuroni
        x = nn.relu(x)        # Funzione di attivazione ReLU
        x = nn.Dense(10)(x)   # Strato denso con 10 output (classi)
        return x

# 3. Inizializzazione dello Stato
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 28 * 28]))["params"]
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# 4. Funzione di Perdita
@jax.jit
def compute_loss(params, apply_fn, x, y):
    logits = apply_fn({"params": params}, x)  # Calcolo dei logits
    one_hot = jax.nn.one_hot(y, num_classes=10)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

# 5. Passo di Aggiornamento
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, x, y)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 6. Funzione di Training
def train_model(state, x_train, y_train, batch_size, epochs):
    num_batches = len(x_train) // batch_size
    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]
        epoch_loss = 0
        for i in range(num_batches):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            state, loss = train_step(state, batch_x, batch_y)
            epoch_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / num_batches:.4f}")
    return state

# 7. Valutazione
@jax.jit
def evaluate(state, x_test, y_test):
    logits = state.apply_fn({"params": state.params}, x_test)
    predictions = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(predictions == y_test)
    return accuracy

# 8. Esecuzione
rng = jax.random.PRNGKey(0)
model = SimpleNN()
state = create_train_state(rng, model, learning_rate=0.001)

state = train_model(state, x_train, y_train, batch_size=64, epochs=5)
accuracy = evaluate(state, x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

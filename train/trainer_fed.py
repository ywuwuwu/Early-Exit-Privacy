
import tensorflow as tf
import numpy as np


def cross_entropy_loss(y_true,exits):
    if len(exits) == 2:
        scalar = [0.4, 0.6]
    if len(exits) == 3:
        scalar = [0.1, 0.1, 0.8]
    total_loss = 0
    for i, exit in enumerate(exits):
        loss_early = tf.keras.losses.categorical_crossentropy(y_true, exit, from_logits = True)
        total_loss += loss_early*scalar[i]

    return total_loss

# Train the model

# Define client update function
def client_update(model, x, y, lr):


    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(x, y, epochs=1, verbose=1)

    weights = model.get_weights()
    loss = history.history['loss'][-1]
    accuracy = np.mean(history.history['accuracy'])
    return weights, loss, accuracy

# Define server aggregation function


def server_aggregate(global_model_weights, local_models_weights, num_clients):
    # Compute the weighted average of the local model weights
    global_weights = []
    for i in range(len(global_model_weights)):
        weight_sum = np.zeros_like(global_model_weights[i])
        for j in range(num_clients):
            weight_sum += local_models_weights[j][i]
        global_weight = weight_sum / num_clients
        global_weights.append(global_weight)
    return global_weights



def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

#global_model_inference, 
#model with early exit

# input mode is branchy_AlexNet or branchy_CNNNet or branch_Resnet 
# optimizers are a set of intance of the optimizer for each model
def training_loop(client_data, num_clients, x_test,y_test,callback, input_model, global_model,model_inference,optimizers, metric, metric_test, batch_size, total_epochs, lr, args):
    
    for epoch in range(total_epochs):
        print("Round {}/{}".format(i+1, total_epochs))    
        local_model_weight = []
        loss_locals = []
        train_loss = 0
        test_loss = 0

        # Update the learning rate using the scheduler
        # optimizer.learning_rate = scheduler(epoch, lr)
        # Reset the accuracy metric
        metric.reset_states()
        metric_test.reset_states()
        global_weights = global_model.get_weights()
# train
        for j in range(num_clients):
            # Get client data
            x, y = client_data[j]
            metric.reset_states()
            # print('x = ', np.shape(x))
            # print('y = ', np.shape(y))
            y = np.argmax(y, axis=-1)
            # Perform local training on client data and get updated model weights
            local_model = input_model(args) # input model 
            local_model.set_weights(global_weights)
            optimizer = optimizers[j]
            # weights, loss, acc = client_update(local_model, x, y, learning_rate)
            for batch in range(0, len(x), batch_size):
                x_batch = x[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]
                with tf.GradientTape() as tape:
                    exits_logits = local_model(x_batch)
                    losses =  cross_entropy_loss(y_batch,exits_logits)
                grads = tape.gradient(losses, local_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, local_model.trainable_variables))
                # Convert the one-hot encoded labels to integer labels
                y_batch = tf.argmax(y_batch, axis=1)
                metric.update_state(y_batch, exits_logits[-1])    

            # Compute the total loss
            for loss in losses:
                train_loss += loss

            local_model_weight.append(local_model.get_weights())
            loss_locals.append(loss)
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(i+1, loss_avg))

        global_weights = server_aggregate(global_weights, local_model_weight, num_clients)
        global_model.set_weights(global_weights)
            # Print the results
        #### This should be a validation/test for the loss value need midify
        # Collect the required logs in a dictionary
        # change it to the global model early exit model
        # 
        # test
        for batch in range(0, len(x_test), batch_size):
            x_test_batch = x_test[batch:batch+batch_size]
            y_test_batch = y_test[batch:batch+batch_size]
            branch_output_test = model_inference(x_test_batch) # global inference
            test_loss0 = tf.keras.losses.categorical_crossentropy(y_test_batch, branch_output_test,  from_logits = True)
            for loss in test_loss0:
                test_loss += loss
            y_test_batch = tf.argmax(y_test_batch, axis=1)
            metric_test.update_state(y_test_batch, branch_output_test)    

        logs = {'loss': test_loss.numpy()/ len(x_test), 'val_accuracy': metric_test.result().numpy()}

        # Call the on_epoch_end method with the logs dictionary
        callback.on_epoch_end(epoch, logs=logs)
        print("Epoch:", epoch, "train_Loss:", loss_avg, "train_accuracy:", metric.result().numpy())
        print("Epoch:", epoch, "test_Loss:", test_loss.numpy()/len(x_test), "test_accuracy:", metric_test.result().numpy())

    return callback
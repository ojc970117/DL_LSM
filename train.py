import tensorflow as tf
from loss.loss import *
from model.simclr_model import *
import numpy as np

def pretrain(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model = "SimCLR", seed=42):
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32)  # (1000, 8, 8, 20)
    data_2 = tf.cast(data_2, tf.float32)  # (1000, 8, 8, 20)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size
    
    # 모델 초기화
    model = build_simclr_model(pre_model = pre_model)
    # model = build_cnn_model(input_shape=data_1.shape[1:])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 학습 스텝 정의
    def train_step(pair_1, pair_2):
        with tf.GradientTape() as tape:
            z1 = model(pair_1, training=True)
            z2 = model(pair_2, training=True)
            loss = nt_xent(z1, z2)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    data = np.concatenate((data_1, data_2), axis=3)

    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]
            
            loss = train_step(batch_data[:,:,:,:20],batch_data[:,:,:,20:])
            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return model
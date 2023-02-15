const tf = require('@tensorflow/tfjs-node');
async function go() {
const model = tf.sequential();  //  последовательность слоев
model.add(tf.layers.dense({units: 2, activation: 'sigmoid', inputShape: [2]}));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});     // meanSquaredError - среднеквадратическая ошибка; sgd - стохастический градиентный спуск

const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
const target_data = tf.tensor2d([[0],[1],[1],[0]]);
  
await model.fit(training_data, target_data,
    { epochs: 100,
      verbose: false,
        callbacks: {
            onEpochEnd: async(epochs, logs) => {
              console.log("Эпоха: " + ((epochs)+1) + " Потери: " + logs.loss);
            }
        }
    }
);
model.predict(training_data).print()
}
go();

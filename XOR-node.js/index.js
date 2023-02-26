const tf = require('@tensorflow/tfjs-node');
async function go() {

const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
const target_data = tf.tensor2d([[0],[1],[1],[0]]);

const model = tf.sequential();  //  последовательность слоев
model.add(tf.layers.dense({units: 2, activation: 'sigmoid', inputShape: [2]}));    // units - количество нейронов;  inputShape - форма ожидаемых данных
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));                   // так как у нас каждый входной экземпляр представлен вектором из двух значений X1 и X2, поэтому inputShape=[2]

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});     // meanSquaredError - среднеквадратическая ошибка; sgd - стохастический градиентный спуск
  
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
model.predict(tf.tensor2d([1,0], [1,2])).print()
}
go();
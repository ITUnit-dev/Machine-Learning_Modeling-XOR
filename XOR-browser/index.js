async function go() {
  console.clear()
  let units_input = Number(document.querySelector("#units-input-value").value)
  let units_output = Number(document.querySelector("#units-output-value").value)
  let count_epohs = Number(document.querySelector("#units-epohs-value").value)
  let messages = document.querySelector(".messages")

  messages.innerHTML = ""  // необходимо пофиксить процесс отображения
  const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
  const target_data = tf.tensor2d([[0],[1],[1],[0]]);

  const model = tf.sequential();  //  последовательность слоев
  model.add(tf.layers.dense({units: units_input, activation: 'sigmoid', inputShape: [2]}));    // units - количество нейронов;  inputShape - форма ожидаемых данных
  model.add(tf.layers.dense({ units: units_output, activation: 'sigmoid' }));                   // так как у нас каждый входной экземпляр представлен вектором из двух значений X1 и X2, поэтому inputShape=[2]

  const surface = { name: 'Model Summary', tab: 'Model Inspection'};
  tfvis.show.modelSummary(surface, model);
  const surface1 = { name: 'Layer Summary', tab: 'Model Inspection'};
  tfvis.show.layer(surface1, model.getLayer(undefined, 1));
  const surface2 = {name: 'Values Distribution_training_data', tab: 'Values Distribution Inspection'};
  await tfvis.show.valuesDistribution(surface2, training_data);
  const surface3 = {name: 'Values Distribution_target_data', tab: 'Values Distribution Inspection'};
  await tfvis.show.valuesDistribution(surface3, target_data);
  
  const surface4 = { name: 'show.fitCallbacks', tab: 'Training' };


  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});     // meanSquaredError - среднеквадратическая ошибка; sgd - стохастический градиентный спуск
    
  await model.fit(training_data, target_data,
      { epochs: count_epohs,
        verbose: false,
          callbacks: {
              onEpochEnd: async(epochs, logs) => {
                console.log("------------------------------------------------")
                // model.layers[0].getWeights()[0].print()
                // model.layers[0].getWeights()[1].print()
                //console.log(model.getWeights().length)
                for (let i = 0; i < model.getWeights().length; i++) {
                  model.getWeights()[i].print()
              }
                let p = document.createElement("p")
                //console.log("Эпоха: " + ((epochs)+1) + " Потери: " + logs.loss)
                p.textContent = `Эпоха: ${((epochs)+1)} Потери: ${logs.loss}`
                messages.append(p)
              }
          }
      }
  );
}
go()
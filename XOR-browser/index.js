async function go() {

  console.clear() // очищаем консоль браузера

  let units_input = Number(document.querySelector("#units-input-value").value)
  let units_output = Number(document.querySelector("#units-output-value").value)
  let count_epohs = Number(document.querySelector("#units-epohs-value").value)
  let save_model = document.querySelector("#units-save-value")
  let batchSize_value = Number(document.querySelector("#units-batchSize-value").value)

  let select_form_activation = document.querySelector(".count-units-activation-list").value
  let select_form_optimizer = document.querySelector(".count-units-optimizer-list").value
  let select_form_loss = document.querySelector(".count-units-loss-list").value

  let messages = document.querySelector(".messages")
  
  let weight_1_1_1 = document.querySelector(".weight_1_1_1_change")
  let weight_1_1_2 = document.querySelector(".weight_1_1_2_change")
  let weight_1_2_1 = document.querySelector(".weight_1_2_1_change")
  let weight_1_2_2 = document.querySelector(".weight_1_2_2_change")
  let weight_2_1_1 = document.querySelector(".weight_2_1_1_change")
  let weight_2_1_2 = document.querySelector(".weight_2_1_2_change")

  let bias_1_1_1 = document.querySelector(".bias_1_1_1_change")
  let bias_1_1_2 = document.querySelector(".bias_1_1_2_change")
  let bias_2_1_1 = document.querySelector(".bias_2_1_1_change")

  messages.innerHTML = ""  // необходимо пофиксить процесс отображения       <!--   !!!   -->
  const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
  const target_data = tf.tensor2d([[0],[1],[1],[0]]);
  const model = tf.sequential();  //  последовательность слоев
  model.add(tf.layers.dense({units: units_input, activation: select_form_activation, inputShape: [2]}));    // units - количество нейронов;  inputShape - форма ожидаемых данных
  model.add(tf.layers.dense({units: units_output, activation: select_form_activation}));                   // так как у нас каждый входной экземпляр представлен вектором из двух значений X1 и X2, поэтому inputShape=[2]
  // Этот код необходим для работы и корректного отбражения tf.js-vis
  const surface = { name: 'Model Summary', tab: 'Model Inspection'};
  tfvis.show.modelSummary(surface, model);
  const surface1 = { name: 'Layer Summary', tab: 'Model Inspection'};
  tfvis.show.layer(surface1, model.getLayer(undefined, 1));
  const surface2 = {name: 'Values Distribution_training_data', tab: 'Values Distribution Inspection'};
  await tfvis.show.valuesDistribution(surface2, training_data);
  const surface3 = {name: 'Values Distribution_target_data', tab: 'Values Distribution Inspection'};
  await tfvis.show.valuesDistribution(surface3, target_data);
  const surface4 = { name: 'show.fitCallbacks', tab: 'Training' };
  // ----------------------------------------------------------------
  
  model.compile({loss: select_form_loss, optimizer: select_form_optimizer});     // meanSquaredError - среднеквадратическая ошибка; sgd - стохастический градиентный спуск
  
  await model.fit(training_data, target_data,
      { epochs: count_epohs,
        batchSize: batchSize_value,
        verbose: false,
          callbacks: [{
              onEpochEnd: async(epochs, logs) => {
                weight_1_1_1.innerText = model.getWeights()[0].dataSync()[0]
                weight_1_1_2.innerText = model.getWeights()[0].dataSync()[1]
                weight_1_2_1.innerText = model.getWeights()[0].dataSync()[2]
                weight_1_2_2.innerText = model.getWeights()[0].dataSync()[3]

                bias_1_1_1.innerText = model.getWeights()[1].dataSync()[0]
                bias_1_1_2.innerText = model.getWeights()[1].dataSync()[1]

                weight_2_1_1.innerText = model.getWeights()[2].dataSync()[0]
                weight_2_1_2.innerText = model.getWeights()[2].dataSync()[1]

                bias_2_1_1.innerText = model.getWeights()[3].dataSync()[0]

                let p = document.createElement("p")
                p.textContent = `Эпоха: ${((epochs)+1)} Потери: ${logs.loss}`
                messages.append(p)
              }
          },
          tfvis.show.fitCallbacks(surface4, ['loss', 'acc'])]
      }
  )
  if(save_model.checked) {
    await model.save('downloads://');
  }
}
go()

let load_model = async () => {
  let uploadJSONInput = document.getElementById("upload-json").value
  let uploadWeightsInput = document.getElementById("upload-weights").value
  if (uploadJSONInput.indexOf(".json") != -1 && uploadWeightsInput.indexOf(".weights.bin") != -1) {

  }
}

let predict = async () => {
  let uploadJSONInput = document.getElementById("upload-json")
  let uploadWeightsInput = document.getElementById("upload-weights")
  let test_value_1 = document.querySelector(".test-model-value-1-input")
  let test_value_2 = document.querySelector(".test-model-value-2-input")
  let print_result = document.querySelector(".print-result")
  const model = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]))
  let result = model.predict(tf.tensor2d([[test_value_1,test_value_2]])).arraySync()[0][0].toFixed(3)
  print_result.innerText = "Полученное значение: " + result;
}

let submit_save = document.querySelector(".load")
submit_save.addEventListener("click", () => {
  submit_save.style.backgroundColor = "green"
  submit_save.style.color = "azure"
  submit_save.innerText = "Загружено!"
}, false)
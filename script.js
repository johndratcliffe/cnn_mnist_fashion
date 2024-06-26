import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js'

const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

const INPUTS = TRAINING_DATA.inputs
const OUTPUTS = TRAINING_DATA.outputs
tf.util.shuffleCombo(INPUTS, OUTPUTS)

function normalize (tensor, min, max) {
  return tf.tidy(function () {
    const MIN_VALUE = tf.scalar(min)
    const MAX_VALUE = tf.scalar(max)

    const RANGE = tf.sub(MAX_VALUE, MIN_VALUE)
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUE)
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE)

    return NORMALIZED_VALUES
  })
}

const INPUTS_TENSOR = tf.tidy(function () { return normalize(tf.tensor2d(INPUTS), 0, 255) })
const OUTPUTS_TENSOR = tf.tidy(function () { return tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10) })

const model = tf.sequential()

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 16,
  kernelSize: 2,
  strides: 1,
  padding: 'same',
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 2,
  strides: 1,
  padding: 'same',
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
model.add(tf.layers.flatten())
model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

model.summary()

train()

async function train () {
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const RESHAPED_TENSOR = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1])
  await model.fit(RESHAPED_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.15,
    epochs: 5,
    batchSize: 256,
    callbacks: { onEpochEnd: logProgress }
  })
  RESHAPED_TENSOR.dispose()
  INPUTS_TENSOR.dispose()
  OUTPUTS_TENSOR.dispose()

  evaluate()
}

function logProgress (epoch, logs) {
  console.log('Epoch: ' + epoch, logs)
}

const PREDICTION_ELEMENT = document.getElementById('prediction')
function evaluate () {
  const OFFSET = Math.floor((Math.random() * INPUTS.length))
  const answer = tf.tidy(function () {
    const newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255)
    const output = model.predict(newInput.reshape([1, 28, 28, 1]))
    output.print()
    return output.squeeze().argMax()
  })

  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = LOOKUP[index]
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong')
    answer.dispose()
    drawImage(INPUTS[OFFSET])
  })
}

let interval = 1000
const RANGER = document.getElementById('ranger')
const DOM_SPEED = document.getElementById('domSpeed')
RANGER.addEventListener('input', function (e) {
  interval = this.value
  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms'
})

const CANVAS = document.getElementById('canvas')
const CTX = CANVAS.getContext('2d')
function drawImage (item) {
  const imageData = CTX.getImageData(0, 0, 28, 28)
  for (let i = 0; i < item.length; i++) {
    imageData.data[i * 4] = item[i] * 255
    imageData.data[i * 4 + 1] = item[i] * 255
    imageData.data[i * 4 + 2] = item[i] * 255
    imageData.data[i * 4 + 3] = 255
  }
  CTX.putImageData(imageData, 0, 0)
  setTimeout(evaluate, interval)
}

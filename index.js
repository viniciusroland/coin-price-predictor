const fetch = require('node-fetch')
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const graphicInputs = []
function formatDate(date) {
    var d = new Date(date),
        month = '' + (d.getMonth() + 1),
        day = '' + d.getDate(),
        year = d.getFullYear();

    if (month.length < 2) month = '0' + month;
    if (day.length < 2) day = '0' + day;

    return [year, month, day].join('-');
}
function getDateArray(start, end) {
  let arr = new Array()
  let dt = new Date(start)
  while(dt <= end) {
    arr.push(formatDate(new Date(dt)))
    dt.setDate(dt.getDate() + 1)
  }
  return arr
}
function setUrl(date) {
  return 'https://api.exchangeratesapi.io/' + date + '?base=USD&symbols=BRL,USD'
}
async function getCoinPrices(start, end) {
  let dateArray = getDateArray(start, end)
  let dolarPriceAndDate = []
  for(let date of dateArray) {
    const res = await fetch(setUrl(date))
    const json = await res.json()
    if(json.date == date) {
      const dolarPrice = json.rates.BRL
      dolarPriceAndDate.push({
        price : dolarPrice,
        date : date
      })
    }
  }
  return dolarPriceAndDate
}
const timePortion = 7
async function createTrainData(data) {
  let trainXs = []
  let trainYs = []
  for (let i = timePortion; i < data.length; i++) {
    let trainX = []
    for (let j = (i - timePortion); j < i; j++) {
      trainX.push([data[j].price]);
    }
    trainXs.push(trainX)
    trainYs.push([data[i].price])
  }
  return {
    trainX : trainXs,
    trainY : trainYs
  }
}


async function createModel() {
  const model = tf.sequential()
  model.add(tf.layers.inputLayer({
    inputShape : [7, 1]
  }))
  model.add(tf.layers.conv1d({
    kernelSize: 2,
    filters: 256,
    strides: 1,
    use_bias: true,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
  }))
  model.add(tf.layers.averagePooling1d({
    poolSize: [2],
    strides: [1]
  }))
  model.add(tf.layers.conv1d({
    kernelSize: 2,
    filters: 128,
    strides: 1,
    use_bias: true,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
  }))
  model.add(tf.layers.averagePooling1d({
    poolSize: [2],
    strides: [1]
  }))
  model.add(tf.layers.flatten({}))
  model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'VarianceScaling',
    activation: 'linear'
  }))
  return model
}
async function trainModel(model, xs, ys) {
  for(let i = 0; i < 35; i++) {
    const response = await model.fit(xs, ys, {epochs : 80, shuffle : true})
  }
}

async function getActualPrice(date) {
  const res = await fetch(setUrl(date))
  const json = await res.json()
  const dolarPrice = json.rates.BRL
  return dolarPrice
}

async function getXsToPredict(data, timePortion) {
    let size = data.length;
    let xs = [];
    for (let i = (size - timePortion); i < size; i++) {
        xs.push([data[i].price]);
    }
    return xs;
}
let start_date_training = new Date('2018-01-18')
let end_date_training = new Date('2019-01-18')

//predicting the next 6 months
let start_date_testing = new Date('2019-01-18')
let end_date_testing = new Date('2019-07-19')

const test_dates = getDateArray(start_date_testing, end_date_testing)
const test_dates_length = test_dates.length
async function testModel(data, model, date, counter) {
  const testXs = await getXsToPredict(data, timePortion)
  const tensorTestXs = tf.tensor3d([testXs])
  const real = await getActualPrice(date)
  const result = model.predict(tensorTestXs)
  const predict = Array.from(result.dataSync())[0]
  let t = parseInt(counter % 15)
  if(t == 0 || t == 1 || t == 2) {
    data.push({
      //price : answer, putting the real data into the 'real database'
      price : real, // putting the predicted value into the 'real database'
      date : date
    })
  } else {
    data.push({
      //price : answer, putting the real data into the 'real database'
      price : predict, // putting the predicted value into the 'real database'
      date : date
    })

  }
  graphicInputs.push({
    date : date,
    realValue : real,
    predictedValue : predict
  })
}

// one year of data
async function main() {
  const dolarPriceAndDate = await getCoinPrices(start_date_training, end_date_training)
  const trainDataSet = await createTrainData(dolarPriceAndDate)
  const xs = tf.tensor3d(trainDataSet.trainX)
  const ys = tf.tensor2d(trainDataSet.trainY)
  const model = await createModel()
  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
  })
  trainModel(model, xs, ys).then(async () => {
    console.log('ready to test it!')
    for(let i = 0; i < test_dates.length; i++) {
      console.log(test_dates[i])
      await testModel(dolarPriceAndDate, model, test_dates[i], i)
    }
    //for(let date of test_dates) {
    //  console.log(date)
    //  await testModel(dolarPriceAndDate, model, date)
    //}
    let graphicJson = JSON.stringify(graphicInputs);
    fs.writeFile('graphic-inputs-test5.json', graphicJson, 'utf8', function(){console.log('dados salvos!')})
  })
}
main()

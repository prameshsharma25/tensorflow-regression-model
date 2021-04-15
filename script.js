const getData = async () => {
    const carsDataResponse =  (await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json"));
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
       mpg: car.Miles_per_Gallon,
       horsepower: car.Horsepower, 
    }))
    .filter(car => car.mpg != null && car.horsepower != null);

    return cleaned;
}

const run = async () => {
    // get car data and map it
    const data = await getData();
    const values = data.map(d => ({
        x: d.mpg,
        y: d.horsepower,
    }));

    // scatterplote of regression
    tfvis.render.scatterplot(
        {name: "Horspower vs. MPG"},
        {values},
        {
            xLabel: "Horsepower",
            yLablel: "MPG",
            height: 300
        }
    );

    // instantiate our model and visualize it
    const model = createModel();
    tfvis.show.modelSummary({name: "Model Summary"}, model);

    // convert data for training
    const tensorData = convertToTensor(data); 
    const {inputs, labels} = tensorData;

    // train the model
    await trainModel(model, inputs, labels);
    console.log("Done Training");
}

const createModel = () => {
    // instantiate a tf model
    // the model is sequential becuase its input flows straight to its output
    const model = tf.sequential();

    // define an input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // define the output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}

const convertToTensor = (data) => {
    return tf.tidy(() => {
        // shuffle data
        tf.util.shuffle(data);

        // convert data to tensors
        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // normalize data
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        // 0 to 1 range
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        };
    });
}

const trainModel = async (model, inputs, labels) => {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    });
}



document.addEventListener("DOMContentLoaded", run);

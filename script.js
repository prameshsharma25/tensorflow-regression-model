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

document.addEventListener("DOMContentLoaded", run);

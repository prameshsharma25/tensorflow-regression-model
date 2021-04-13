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
    const data = await getData();
    const values = data.map(d => ({
        x: d.mpg,
        y: d.horsepower,
    }));

tfvis.render.scatterplot(
    {name: "Horspower vs. MPG"},
    {values},
    {
        xLabel: "Horsepower",
        yLablel: "MPG",
        height: 300
    }
);
}

document.addEventListener("DOMContentLoaded", run);
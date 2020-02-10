Dependencies - node
npm install connect serve-static

This project offers a web interface to the MNIST digit recognition model which is tensorflow based.
Follow the steps to fire it up and enjoy..

1) cd webmnist-frontend; node serve.js 
This command serves up the html UI on port 8081

2) ./startMNISTDemo.sh
This command fires up the python backend and loads up the trained tensorflow model trained on the mnist dataset.

3) Open up a browser and type "http://localhost:8181"
This shows up an html canvas where a digit can be drawn.

4) Press submit for the prediction to show up.


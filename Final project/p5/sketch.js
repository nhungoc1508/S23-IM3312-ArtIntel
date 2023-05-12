let cnv;
let capture;
let url = "https://10.224.19.171:8000/send-image";
let img;
let nextImg;
let responseReceived = true;
let response;
let responseList = [];
let videoFrame;
let count = 0;
let timestamp = 0;

let x0 = -7.25; // 255 in 34 frames -> 255/34 per frame
let tint_0 = 255;
let x1 = 7.25;
let tint_1 = 0;

function setup() {
  cnv = createCanvas(displayWidth, displayHeight);
  capture = createCapture(VIDEO);
  capture.size(300, 225);
  capture.hide();
}

function draw() {
  //show live webcam feed
  push();
  scale(-1, 1);
  translate(-capture.width, 0);
  image(capture, 0, 0, 300, 225);
  pop();
  
  if (responseReceived) {
    count++;
    console.log("Smile! " + str(count));
    captureFrame();
  }
  
  if (img) {
    tint_0 += x0;
    push();
    tint(255, tint_0);
    imageMode(CENTER);
    // OG: 300:225, to: x:windowHeight -> x/300 = windowHeight/225
    // image(img, windowWidth/2, windowHeight/2, windowHeight/225 * 300, windowHeight);
    image(img, displayWidth/2, displayHeight/2, displayWidth, displayWidth / 300 * 225);
    pop();
    // if (tint_0 <= 0 | tint_0 >= 255) {
    //   x0 = x0 * -1;
    // }
    // if (tint_0 == 0) {
    //   tint_0 = 255;
    // }
  }
  
  if (nextImg) {
    tint_1 += x1;
    push();
    tint(255, tint_1);
    imageMode(CENTER);
    // OG: 300:225, to: x:windowHeight -> x/300 = windowHeight/225
    // Alternative: 300:225 = height:x -> x/225 = height/300
    // image(img, windowWidth/2, windowHeight/2, windowHeight/225 * 300, windowHeight);
    // image(nextImg, displayWidth/2, displayHeight/2, displayHeight/225 * 300, displayHeight);
    image(nextImg, displayWidth/2, displayHeight/2, displayWidth, displayWidth / 300 * 225);
    pop();
    // if (tint_1 <= 0 | tint_1 >= 255) {
    //   x1 = x1 * -1;
    // }
  }
  
  if (videoFrame) {
    // image(videoFrame, 300, 225, 300, 225);
    // image(videoFrame, 0, 0, 300, 225);
  }
  
  if (frameCount % 34 == 0 && responseList.length > 0) {
    let firstElem = responseList.shift();
    if (responseList.length == 1) {
      img = loadImage(firstElem);
    } else {
      img = nextImg;
      nextImg = loadImage(firstElem);
      tint_0 = 255;
      tint_1 = 0;
    }
  }
  
  //show live webcam feed
  push();
  scale(-1, 1);
  translate(-capture.width, 0);
  image(capture, 0, 0, 300, 225);
  pop();
}


function captureFrame() {
  responseReceived = false;
  videoFrame = get(capture.elt.offsetLeft, capture.elt.offsetTop, capture.width, capture.height);
  let imageBase64String = videoFrame.canvas.toDataURL();
  let postData = {title: "capture", image_str: imageBase64String};
  httpPost(url, 'json', postData, function(result) {
    // console.log(Object.keys(result));
    // console.log(Object.keys(result).length);
    // img = loadImage(result.data);
    response = result;
    console.log('Got a response!');
    console.log((frameCount - timestamp) / 60);
    timestamp = frameCount;
    // responseList = Object.entries(response);
    for (const [key, value] of Object.entries(response)) {
      responseList.push(value);
      // console.log(key);
    }
    responseReceived = true;
  })
}

function mousePressed() {
  let fs = fullscreen();
  fullscreen(!fs);
}
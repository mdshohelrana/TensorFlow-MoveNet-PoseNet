import './App.css';

import {useEffect, useState} from "react";
import * as tfJsCore from '@tensorflow/tfjs-core';
import * as tfJsConverter from '@tensorflow/tfjs-converter';
import * as tfJsBackendWebgl from '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import model1 from "./assets/img/model1.jpg";

function App() {
    const [keyPoints, setKeyPoints] = useState();
    useEffect(() => {
        async function fetchData() {
            const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,
                {
                    modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
                });

            const video = document.getElementById('video');
            const poses = await detector.estimatePoses(video);
            console.log(poses[0].keypoints);
            setKeyPoints(JSON.stringify(poses[0].keypoints));
        }

        fetchData();
    }, []); // Or [] if effect doesn't need props or state

    return (
        <div className="App">
            <header className="App-header">
                <img id="video"
                     crossOrigin='anonymous'
                     src={model1}/>
                {keyPoints}
            </header>
        </div>
    );
}

export default App;

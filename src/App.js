import './App.css';
// eslint-disable-next-line
import * as tfJsCore from '@tensorflow/tfjs-core';
// eslint-disable-next-line
import * as tfJsConverter from '@tensorflow/tfjs-converter';
// eslint-disable-next-line
import * as tfJsBackendWebgl from '@tensorflow/tfjs-backend-webgl';

import {useEffect, useState} from "react";
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

        // eslint-disable-next-line
        // noinspection JSIgnoredPromiseFromCall
        fetchData();
    }, []); // Or [] if effect doesn't need props or state

    return (
        <div className="App">
            <header className="App-header">
                <img alt="exercise" id="video"
                     crossOrigin='anonymous'
                     src={model1}/>
                {keyPoints}
            </header>
        </div>
    );
}

export default App;

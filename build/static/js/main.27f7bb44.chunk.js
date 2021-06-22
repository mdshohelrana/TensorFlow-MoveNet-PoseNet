(this["webpackJsonptensor-flow-move-net-pose-net"]=this["webpackJsonptensor-flow-move-net-pose-net"]||[]).push([[0],{248:function(e,t,n){},250:function(e,t,n){},255:function(e,t){},256:function(e,t){},264:function(e,t){},268:function(e,t,n){"use strict";n.r(t);var r,a,i,c,o,s,u,l=n(46),f=n.n(l),d=n(208),h=n.n(d),v=(n(248),n(27)),p=n(9),b=n.n(p),m=n(12),y=n(6),j=(n(250),n(0),n(71),n(267),n(228)),g=n.n(j),x=n(104),O=n(95),w=n(55),k={facingMode:"user",deviceId:"",frameRate:{max:60,ideal:30},width:x.isMobile?360:1280,height:x.isMobile?270:720},S=1,I=.3,M=0,P=0,E=0;var N=function(){var e=Object(l.useState)(!1),t=Object(y.a)(e,2),n=t[0],f=t[1],d=Object(l.useState)(0),h=Object(y.a)(d,2),p=h[0],j=h[1],N=Object(l.useRef)({});Object(l.useEffect)((function(){W().then()}),[]);var W=function(){var e=Object(m.a)(b.a.mark((function e(){return b.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return i&&(window.cancelAnimationFrame(i),r.dispose()),e.next=3,C();case 3:return r=e.sent,e.next=6,T();case 6:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}(),C=function(){var e=Object(m.a)(b.a.mark((function e(){return b.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return s=O.a.MoveNet,u=O.c.modelType.SINGLEPOSE_LIGHTNING,e.next=4,O.b(s,{modelType:u});case 4:return e.abrupt("return",e.sent);case 5:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}(),T=function(){var e=Object(m.a)(b.a.mark((function e(){return b.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,F();case 2:i=requestAnimationFrame(T);case 3:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}(),F=function(){var e=Object(m.a)(b.a.mark((function e(){var t,a;return b.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(n){e.next=3;break}return e.next=3,new Promise((function(e){e(!0)}));case 3:return R(),t=N.current&&N.current.video,e.next=7,r.estimatePoses(t,{maxPoses:S,flipHorizontal:!1});case 7:a=e.sent,A(),B(t),a.length>0&&D(a);case 11:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}(),R=function(){a=(performance||Date).now()},A=function(){var e=(performance||Date).now();P+=e-a,++M;if(e-E>=1e3){var t=P/M;P=0,M=0,j(1e3/t,120),E=e}},B=function(e){c=document.getElementById("output-full-screen"),o=c.getContext("2d");var t=e.videoWidth,n=e.videoHeight;e.width=t,e.height=n,c.width=t,c.height=n,o.translate(e.videoWidth,0),o.scale(-1,1),o.drawImage(e,0,0,t,n)},D=function(e){var t,n=Object(v.a)(e);try{for(n.s();!(t=n.n()).done;){var r=t.value;G(r)}}catch(a){n.e(a)}finally{n.f()}},G=function(e){null!=e.keypoints&&(L(e.keypoints),J(e.keypoints))},L=function(e){var t=O.d.getKeypointIndexBySide(s);o.fillStyle="White",o.strokeStyle="White",o.lineWidth=2;var n,r=Object(v.a)(t.middle);try{for(r.s();!(n=r.n()).done;){var a=n.value;H(e[a])}}catch(h){r.e(h)}finally{r.f()}o.fillStyle="Green";var i,c=Object(v.a)(t.left);try{for(c.s();!(i=c.n()).done;){var u=i.value;H(e[u])}}catch(h){c.e(h)}finally{c.f()}o.fillStyle="Orange";var l,f=Object(v.a)(t.right);try{for(f.s();!(l=f.n()).done;){var d=l.value;H(e[d])}}catch(h){f.e(h)}finally{f.f()}},H=function(e){if((null!=e.score?e.score:1)>=(I||0)){var t=new Path2D;t.arc(e.x,e.y,4,0,2*Math.PI),o.fill(t),o.stroke(t)}},J=function(e){o.fillStyle="White",o.strokeStyle="White",o.lineWidth=2,O.d.getAdjacentPairs(s).forEach((function(t){var n=Object(y.a)(t,2),r=n[0],a=n[1],i=e[r],c=e[a],s=null!=i.score?i.score:1,u=null!=c.score?c.score:1,l=I||0;s>=l&&u>=l&&(o.beginPath(),o.moveTo(i.x,i.y),o.lineTo(c.x,c.y),o.stroke())}))};return Object(w.jsx)("section",{className:"App h-screen w-full flex justify-center items-center bg-green-500",children:Object(w.jsxs)("div",{className:"bg-gray-800",children:[Object(w.jsx)(g.a,{className:"filter blur-lg",ref:N,audio:!1,height:x.isMobile?270:720,width:x.isMobile?360:1280,videoConstraints:k,onUserMediaError:function(){console.log("ERROR in Camera!")},onUserMedia:function(){console.log("Camera loaded!"),f(!0)}}),Object(w.jsx)("canvas",{className:"absolute",id:"output-full-screen"}),Object(w.jsx)("label",{children:p})]})})},W=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,270)).then((function(t){var n=t.getCLS,r=t.getFID,a=t.getFCP,i=t.getLCP,c=t.getTTFB;n(e),r(e),a(e),i(e),c(e)}))};h.a.render(Object(w.jsx)(f.a.StrictMode,{children:Object(w.jsx)(N,{})}),document.getElementById("root")),W()}},[[268,1,2]]]);
//# sourceMappingURL=main.27f7bb44.chunk.js.map
// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDjhgZhg_RQkea8eToc7neR_9ISKCv2xVM",
  authDomain: "medication-schedule-manager.firebaseapp.com",
  projectId: "medication-schedule-manager",
  storageBucket: "medication-schedule-manager.firebasestorage.app",
  messagingSenderId: "960476325330",
  appId: "1:960476325330:web:a31a9236bd352007bbd40b",
  measurementId: "G-QNKJL767SR"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
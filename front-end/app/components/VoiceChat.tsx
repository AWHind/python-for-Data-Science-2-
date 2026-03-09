"use client";

import { useState } from "react";

export default function VoiceChat() {

  const [reply,setReply] = useState("");

  async function sendMessage(message:string){

    const res = await fetch("http://localhost:8000/chat",{
      method:"POST",
      headers:{
        "Content-Type":"application/json"
      },
      body:JSON.stringify({message})
    })

    const data = await res.json()

    setReply(data.reply)
  }

  async function startVoice(){

    const stream = await navigator.mediaDevices.getUserMedia({audio:true})

    const mediaRecorder = new MediaRecorder(stream)

    let chunks:any[] = []

    mediaRecorder.ondataavailable = e => chunks.push(e.data)

    mediaRecorder.onstop = async () => {

      const blob = new Blob(chunks)

      const res = await fetch("http://localhost:8000/voice-chat",{
        method:"POST",
        body:blob
      })

      const data = await res.json()

      setReply(data.text)

      const audio = new Audio(data.audio)

      audio.play()
    }

    mediaRecorder.start()

    setTimeout(()=>mediaRecorder.stop(),4000)
  }

  return (
    <div>

      <button onClick={()=>sendMessage("What is arrhythmia?")}>
        Ask AI
      </button>

      <button onClick={startVoice}>
        🎤 Talk
      </button>

      <p>{reply}</p>

    </div>
  )
}
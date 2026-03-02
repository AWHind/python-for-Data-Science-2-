"use client"

import { useState } from "react"

const API_URL = "http://127.0.0.1:8000"
const EXPECTED_FEATURES = 278

export default function PredictionForm() {

  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async () => {

    setLoading(true)
    setResult(null)

    try {

      // Example: 11 values
      let features: number[] = [
        54, 1, 172, 78, 80, 160, 370, 180, 100, 72, 6
      ]

      // Complete to 279
      while (features.length < EXPECTED_FEATURES) {
        features.push(0)
      }

      console.log("Sending length:", features.length)

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          features: features
        })
      })

      const data = await response.json()
      console.log("Response:", data)

      setResult(data)

    } catch (error) {
      console.error(error)
      alert("Backend error")
    }

    setLoading(false)
  }

  return (
      <div style={{ padding: 40 }}>
        <button onClick={handlePredict}>
          {loading ? "Loading..." : "Predict"}
        </button>

        {result && (
            <div>
              <h3>Prediction: {result.prediction}</h3>
            </div>
        )}
      </div>
  )
}
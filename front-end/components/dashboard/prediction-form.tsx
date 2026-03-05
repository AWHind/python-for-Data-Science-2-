"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertCircle, CheckCircle2, Loader } from "lucide-react"

const API_URL = "http://127.0.0.1:8000"
const EXPECTED_FEATURES = 278

interface ModelInfo {
  name: string
  accuracy: number
  f1_score: number
  run_id: string
}

interface PredictionResult {
  prediction: number
  model_used: string
  confidence: number | null
  model_metrics: any
}

export default function PredictionForm() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState("RandomForest")
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [apiStatus, setApiStatus] = useState<string>("connecting")

  // Load available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true)
        const healthResponse = await fetch(`${API_URL}/health`)
        if (!healthResponse.ok) {
          setApiStatus("disconnected")
          setError("Cannot connect to API")
          return
        }
        
        setApiStatus("connected")
        
        const modelsResponse = await fetch(`${API_URL}/models`)
        if (modelsResponse.ok) {
          const data = await modelsResponse.json()
          setModels(data)
          if (data.length > 0) {
            setSelectedModel(data[0].name)
          }
        }
      } catch (error) {
        console.error("Failed to fetch models:", error)
        setApiStatus("error")
        setError("Failed to load available models")
      } finally {
        setModelsLoading(false)
      }
    }

    fetchModels()
  }, [])

  const generateSampleFeatures = (): number[] => {
    // Generate realistic ECG feature values
    let features: number[] = [
      54, 1, 172, 78, 80, 160, 370, 180, 100, 72, 6,
      ...Array.from({ length: 20 }, () => Math.random() * 100),
      ...Array.from({ length: 50 }, () => Math.random() * 500)
    ]

    while (features.length < EXPECTED_FEATURES) {
      features.push(Math.random() * 50)
    }

    return features.slice(0, EXPECTED_FEATURES)
  }

  const handlePredict = async (modelToUse?: string) => {
    const model = modelToUse || selectedModel
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const features = generateSampleFeatures()

      console.log(`[v0] Predicting with ${model}. Sending ${features.length} features`)

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          features,
          model
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Prediction failed")
      }

      const data: PredictionResult = await response.json()
      console.log("[v0] Prediction result:", data)
      setResult(data)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred"
      console.error("[v0] Prediction error:", errorMessage)
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const downloadReport = async () => {
    try {
      const features = generateSampleFeatures()
      const response = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          features,
          model: selectedModel
        })
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement("a")
        link.href = url
        link.download = "rapport_prediction.pdf"
        link.click()
      }
    } catch (error) {
      console.error("Report download error:", error)
      setError("Failed to generate report")
    }
  }

  return (
    <div className="space-y-6">
      {/* API Status */}
      <Card className="border-blue-200 bg-blue-50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">API Status</CardTitle>
            <div className="flex items-center gap-2">
              {apiStatus === "connected" && (
                <>
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <span className="text-sm text-green-700">Connected</span>
                </>
              )}
              {apiStatus === "connecting" && (
                <>
                  <Loader className="h-4 w-4 animate-spin" />
                  <span className="text-sm text-blue-700">Connecting...</span>
                </>
              )}
              {apiStatus === "disconnected" && (
                <>
                  <div className="h-2 w-2 rounded-full bg-red-500" />
                  <span className="text-sm text-red-700">Disconnected</span>
                </>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Prediction Card */}
      <Card>
        <CardHeader>
          <CardTitle>Model Prediction</CardTitle>
          <CardDescription>Select a model and generate predictions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Select Model</label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.name} value={model.name}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {/* Model Info */}
            {models.find(m => m.name === selectedModel) && (
              <div className="mt-2 text-xs text-gray-600 bg-gray-50 p-2 rounded">
                <p>Accuracy: {(models.find(m => m.name === selectedModel)?.accuracy! * 100).toFixed(2)}%</p>
                <p>F1 Score: {models.find(m => m.name === selectedModel)?.f1_score.toFixed(4)}</p>
              </div>
            )}
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <Button 
              onClick={() => handlePredict()} 
              disabled={loading || modelsLoading}
              className="flex-1"
            >
              {loading ? (
                <>
                  <Loader className="mr-2 h-4 w-4 animate-spin" />
                  Predicting...
                </>
              ) : (
                "Predict"
              )}
            </Button>
            <Button 
              onClick={downloadReport} 
              variant="outline"
              disabled={!result || loading}
            >
              Download Report
            </Button>
          </div>

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Result */}
          {result && (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  <span className="font-semibold text-green-900">Prediction Result</span>
                </div>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700">
                    <span className="font-medium">Model:</span> {result.model_used}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium">Class:</span> {" "}
                    <span className="font-bold text-lg">
                      {result.prediction === 0 ? "Normal (0)" : "Arrhythmia (1)"}
                    </span>
                  </p>
                  {result.confidence && (
                    <p className="text-gray-700">
                      <span className="font-medium">Accuracy:</span> {(result.confidence * 100).toFixed(2)}%
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Models Comparison */}
      {!modelsLoading && models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>All Models</CardTitle>
            <CardDescription>Compare performance across all available models</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="comparison" className="w-full">
              <TabsList>
                <TabsTrigger value="comparison">Comparison</TabsTrigger>
                <TabsTrigger value="predict">Quick Predict</TabsTrigger>
              </TabsList>

              {/* Comparison Tab */}
              <TabsContent value="comparison" className="space-y-4">
                <div className="space-y-2">
                  {models.map((model) => (
                    <div key={model.name} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium text-sm">{model.name}</p>
                        <p className="text-xs text-gray-600">
                          Accuracy: {(model.accuracy * 100).toFixed(2)}% | F1: {model.f1_score.toFixed(4)}
                        </p>
                      </div>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => {
                          setSelectedModel(model.name)
                          handlePredict(model.name)
                        }}
                        disabled={loading}
                      >
                        Test
                      </Button>
                    </div>
                  ))}
                </div>
              </TabsContent>

              {/* Quick Predict Tab */}
              <TabsContent value="predict" className="space-y-3">
                <div className="text-sm text-gray-600 mb-4">
                  Click on any model below to make a prediction with it
                </div>
                {models.map((model) => (
                  <Button
                    key={model.name}
                    onClick={() => {
                      setSelectedModel(model.name)
                      handlePredict(model.name)
                    }}
                    disabled={loading}
                    variant="outline"
                    className="w-full justify-start"
                  >
                    <span>{model.name}</span>
                    <span className="ml-auto text-xs">
                      {(model.accuracy * 100).toFixed(0)}%
                    </span>
                  </Button>
                ))}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

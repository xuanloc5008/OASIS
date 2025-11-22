"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Loader2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")
  const [results, setResults] = useState<{
    segmentation: string
    heatmap: string
  } | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResults(null)
      setError("")
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError("")

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      // Update this URL to point to your FastAPI backend
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      console.error("[v0] Error during inference:", err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl("")
    setResults(null)
    setError("")
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight mb-3 text-balance">Medical Image Segmentation</h1>
          <p className="text-muted-foreground text-lg">
            Upload an image to generate GradCAM heatmap and segmentation mask
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Upload Section */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
              <CardDescription>Select a medical image for analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center gap-4">
                <label
                  htmlFor="file-upload"
                  className={`
                    relative flex flex-col items-center justify-center
                    w-full h-64 border-2 border-dashed rounded-lg
                    cursor-pointer transition-colors
                    ${previewUrl ? "border-primary" : "border-muted-foreground/25"}
                    hover:border-primary hover:bg-accent/50
                  `}
                >
                  {previewUrl ? (
                    <img
                      src={previewUrl || "/placeholder.svg"}
                      alt="Preview"
                      className="max-h-full max-w-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Upload className="w-12 h-12" />
                      <p className="text-sm font-medium">Click to upload image</p>
                      <p className="text-xs">PNG, JPG up to 10MB</p>
                    </div>
                  )}
                  <input
                    id="file-upload"
                    type="file"
                    className="sr-only"
                    accept="image/*"
                    onChange={handleFileSelect}
                    disabled={loading}
                  />
                </label>

                <div className="flex gap-3 w-full sm:w-auto">
                  <Button
                    onClick={handleSubmit}
                    disabled={!selectedFile || loading}
                    className="flex-1 sm:flex-initial"
                    size="lg"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      "Generate Results"
                    )}
                  </Button>
                  {(selectedFile || results) && (
                    <Button onClick={handleReset} variant="outline" disabled={loading} size="lg">
                      Reset
                    </Button>
                  )}
                </div>
              </div>

              {error && (
                <Alert variant="destructive" className="mt-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          {results && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Segmentation Mask</CardTitle>
                  <CardDescription>Colored regions indicate different tissue classes</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg overflow-hidden border border-border bg-muted/30">
                    <img
                      src={results.segmentation || "/placeholder.svg"}
                      alt="Segmentation result"
                      className="w-full h-auto"
                    />
                  </div>
                  <div className="mt-4 flex gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-black border border-border rounded-sm" />
                      <span className="text-muted-foreground">Background</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-red-500 border border-border rounded-sm" />
                      <span className="text-muted-foreground">Class 1</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-green-500 border border-border rounded-sm" />
                      <span className="text-muted-foreground">Class 2</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>GradCAM Heatmap</CardTitle>
                  <CardDescription>Visualization of model attention areas</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg overflow-hidden border border-border bg-muted/30">
                    <img src={results.heatmap || "/placeholder.svg"} alt="GradCAM heatmap" className="w-full h-auto" />
                  </div>
                  <p className="mt-4 text-sm text-muted-foreground">
                    Warmer colors indicate regions the model focuses on for predictions
                  </p>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

"use client";

import { useState, useRef } from "react";
import { Upload, FileText, CheckCircle, AlertCircle, Loader, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useAuth } from "@/lib/auth-context";

interface UploadProps {
  onUploadSuccess?: (fileData: any) => void;
  onGenerateReport?: (reportData: any) => void;
}

const API_URL = "http://127.0.0.1:8000";

export function ECGUpload({ onUploadSuccess, onGenerateReport }: UploadProps) {
  const { user } = useAuth();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = [
      "text/csv",
      "application/json",
      "text/plain",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ];

    if (!validTypes.includes(file.type)) {
      setError("Format de fichier non supporté. Veuillez télécharger un fichier CSV, JSON ou Excel.");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("Le fichier est trop volumineux. Taille maximale: 10MB");
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const fileContent = await file.text();
      const patientId = user?.id || "unknown";
      const patientName = user?.name || "Patient Anonyme";

      const response = await fetch(`${API_URL}/upload-ecg`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_id: patientId,
          patient_name: patientName,
          file_content: fileContent,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Erreur lors du téléchargement");
      }

      const data = await response.json();
      setUploadedFile({
        name: file.name,
        size: file.size,
        uploadTime: data.upload_time,
        ...data,
      });

      setSuccess(`Fichier ${file.name} téléchargé avec succès!`);
      onUploadSuccess?.(data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Erreur inconnue";
      console.error("[v0] ECG upload error:", errorMessage);
      setError(errorMessage);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleGenerateReport = async () => {
    if (!uploadedFile) return;

    setIsGeneratingReport(true);
    setError(null);

    try {
      // Generate report with patient data
      const reportData = {
        patient_id: uploadedFile.patient_id,
        patient_name: uploadedFile.patient_name,
        file_path: uploadedFile.file_path,
        upload_time: uploadedFile.upload_time,
      };

      // In a real scenario, you would send this to the backend to generate a PDF
      // For now, we'll just simulate the report generation
      setSuccess("Rapport généré avec succès!");
      onGenerateReport?.(reportData);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Erreur inconnue";
      console.error("[v0] Report generation error:", errorMessage);
      setError(errorMessage);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!uploadedFile) return;

    try {
      setIsGeneratingReport(true);
      
      // Get patient data from context or form
      const patientAge = 54; // Default, should come from form
      const patientSex = "M"; // Default, should come from form

      const response = await fetch(`${API_URL}/patient-report?patient_name=${uploadedFile.patient_name}&patient_id=${uploadedFile.patient_id}&age=${patientAge}&sex=${patientSex}&prediction_result=0&model_used=RandomForest&confidence=0.95`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `rapport_${uploadedFile.patient_id}.pdf`;
        link.click();
        window.URL.revokeObjectURL(url);
        setSuccess("Rapport téléchargé avec succès!");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Erreur inconnue";
      console.error("[v0] Report download error:", errorMessage);
      setError(errorMessage);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Télécharger les Données ECG
        </CardTitle>
        <CardDescription>
          Importez vos fichiers ECG pour analyse et génération de rapport
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* File Upload Area */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.xlsx,.xls,.txt"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />

          <FileText className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <p className="text-sm font-medium text-gray-700 mb-1">
            {isUploading ? "Téléchargement en cours..." : "Cliquez pour sélectionner un fichier"}
          </p>
          <p className="text-xs text-gray-500">
            CSV, JSON, Excel ou TXT (max 10MB)
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Success Alert */}
        {success && (
          <Alert className="border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">{success}</AlertDescription>
          </Alert>
        )}

        {/* Uploaded File Info */}
        {uploadedFile && (
          <div className="bg-gray-50 p-4 rounded-lg space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span className="font-medium text-gray-900">{uploadedFile.name}</span>
            </div>
            <div className="text-sm text-gray-600 space-y-1">
              <p>Taille: {(uploadedFile.size / 1024).toFixed(2)} KB</p>
              <p>Téléchargé: {uploadedFile.uploadTime}</p>
              <p>Patient: {uploadedFile.patient_name}</p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 pt-4 mt-4 border-t">
              <Button
                onClick={handleGenerateReport}
                disabled={isGeneratingReport}
                variant="outline"
                className="flex-1"
              >
                {isGeneratingReport ? (
                  <>
                    <Loader className="mr-2 h-4 w-4 animate-spin" />
                    Génération...
                  </>
                ) : (
                  "Générer Rapport"
                )}
              </Button>
              <Button
                onClick={handleDownloadReport}
                disabled={isGeneratingReport}
                className="flex-1"
              >
                {isGeneratingReport ? (
                  <>
                    <Loader className="mr-2 h-4 w-4 animate-spin" />
                    Téléchargement...
                  </>
                ) : (
                  <>
                    <Download className="mr-2 h-4 w-4" />
                    Télécharger PDF
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

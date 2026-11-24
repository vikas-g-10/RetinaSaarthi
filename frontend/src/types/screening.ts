export type QualityDecision = 'PASS' | 'RECAPTURE' | 'MANUAL_REVIEW' | string;
export interface QualityMetrics { width: number; height: number; pixel_count: number; sharpness: number; illumination_mean: number; illumination_std: number; illumination_quadrant_ratio: number; contrast_p5_p95: number; field_coverage_percent: number; field_coverage_frame_percent: number; field_touches_edge: boolean; touches_top: boolean; touches_bottom: boolean; touches_left: boolean; touches_right: boolean; }
export interface QualityResponse { project: string; assessment: { decision: QualityDecision; reason: string }; metrics: QualityMetrics | null; clinical_status: string; }
export interface ApiError { status: number | null; title: string; message: string; }
export interface ModelStatus { ready: boolean; model_name: string; model_version: string | null; classes: string[]; error?: string; }
export interface Prediction { predicted_class: number; predicted_label: string; probabilities: number[]; model_name: string; model_version: string; }
export interface ClassificationResponse { project: string; result_id?: string; quality: { decision: QualityDecision; reason: string; metrics: QualityMetrics | null }; prediction: Prediction | null; }

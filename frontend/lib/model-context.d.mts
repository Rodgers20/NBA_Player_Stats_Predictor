import type {GamePrediction} from './types';
export function modelContext(prediction?: GamePrediction): {label:string;warnings:string[];probability:number|null};

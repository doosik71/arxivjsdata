# Thoracic Surgery Video Analysis for Surgical Phase Recognition

Syed Abdul Mateen, Niharika Malvia, Syed Abdul Khader, Danny Wang, Deepti Srinivasan, Chi-Fu Jeffrey Yang, Lana Schumacher, Sandeep Manjanna (2024)

## 🧩 Problem to Solve

본 논문은 흉부 외과(Thoracic Surgery) 수술 비디오 데이터를 활용한 **Surgical Phase Recognition (SPR, 수술 단계 인식)** 문제를 해결하고자 한다. 수술 단계 인식은 현재 진행 중인 수술 시나리오를 인식 및 평가하고, 수술 과정을 요약하며, 외과의의 기술을 평가하고, 의사 결정 지원 및 의료 교육을 촉진할 수 있는 핵심 기술이다.

특히 흉부 외과의 경우, 수술 중 치명적인 사고가 발생하여 개흉술(open thoracotomy)로 전환해야 하는 사례가 보고되고 있다. 따라서 자동화된 워크플로우 분석을 통해 수술 중 실시간 인지 가이드를 제공하거나 교육생에게 정밀한 피드백을 제공함으로써 수술의 안전성과 효율성을 높이는 것이 본 연구의 주요 목표이다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 이미지 기반의 프레임 분석(Frame-based)과 비디오 클립 기반의 분석(Video clipping-based) 방식의 성능을 비교하여, 수술 단계 인식에서 **시계열 정보(Temporal dependencies)**의 중요성을 검증하는 것이다. 특히, 단순한 비디오 모델을 넘어 이미지 및 비디오 교사 모델의 지식을 전이받는 **Masked Video Distillation (MVD)** 기법을 적용하여 흉부 외과 수술 단계 인식 성능을 극대화하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 모델들을 기반으로 실험을 진행하였다.

- **ImageNet ViT (Vision Transformer):** 이미지 기반의 분류 모델로, 비디오의 개별 프레임을 독립적으로 분석하는 기준선(Baseline)으로 사용되었다.
- **VideoMAE (Video Masked Autoencoder):** 비디오 이해를 위해 설계된 모델로, Masked feature modeling을 통해 데이터 효율적인 학습을 수행한다.
- **Masked Video Distillation (MVD):** VideoMAE와 유사한 Transformer 구조를 가지나, 사전 학습된 이미지 및 비디오 교사 모델(Teacher models)로부터 학생 모델(Student encoder)로 지식을 전이하는 Transfer Learning 방식을 도입하여 더 풍부한 표현력을 가진다.

본 논문은 기존의 단순 이미지 분류 방식이 가진 한계, 즉 비디오의 연속적인 흐름(Temporal context)을 무시한다는 점을 지적하며, 이를 해결하기 위해 Video-based 모델들의 효용성을 입증하고자 하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

시스템은 입력 비디오를 짧은 클립으로 나누어 처리하는 구조를 가진다. 전체적인 흐름은 다음과 같다.

1. **데이터 전처리:** 수술 비디오를 10초 길이의 클립으로 분할한다. 이때 Sliding window 방식을 사용하며, 데이터 중복을 피하기 위해 Stride를 10초로 설정하였다.
2. **입력 데이터:** 각 클립은 16개의 프레임으로 구성된다.
3. **특징 추출:**
   - **3D Patch Embedding:** 입력된 16프레임 비디오 클립에 커널 크기가 $[2, 16, 16]$인 3D Convolution을 적용하여 패치 임베딩(Patch Embedding)을 생성한다. 결과물의 형태는 $[\text{batch\_size}, 1568, 768]$이다.
   - **Positional Embedding:** 공간적 정보를 통합하기 위해 위치 임베딩을 추가한다.
   - **Transformer Encoder:** ViT 구조의 인코더를 통해 최종적인 특징(Feature)을 추출한다.

### 2. 모델별 세부 사항

- **ImageNet ViT:** 단일 프레임의 이미지 특성을 기반으로 분류를 수행한다.
- **VideoMAE & MVD:** Kinetics-400 데이터셋으로 사전 학습된 $\text{ViT-L}$ 백본을 사용한다. 두 모델 모두 Masked feature modeling을 사용하지만, MVD는 추가적으로 교사 모델로부터 지식을 전이받는 distillation 과정을 거친다.

### 3. 학습 및 평가 절차

- **학습 설정:** VideoMAE와 MVD 모델을 100 epoch 동안 Fine-tuning 하였으며, $\text{Top-1 Accuracy}$를 모니터링하여 최적의 체크포인트를 선택하였다.
- **데이터 분할:** 총 17명의 환자 데이터셋을 학습 및 검증 세트(13케이스)와 테스트 세트(4케이스)로 분리하였다. 테스트 세트의 데이터는 학습 과정에서 완전히 배제하여 일반화 성능을 엄격히 평가하였다.
- **강건성 확보:** 학습/검증 세트 내에서는 overlapping split 전략을 사용하여 모델의 강건성을 높였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Massachusetts General Hospital(MGH)에서 수집한 17개의 흉부 외과 수술 비디오 (평균 길이 2.18시간).
- **분류 클래스:** 총 11개의 수술 단계(Surgical phases).
- **평가 지표:** $\text{Top-1 Accuracy}$ (가장 확률이 높은 예측값이 정답인 비율)와 $\text{Top-5 Accuracy}$ (상위 5개 예측값 내에 정답이 포함된 비율).

### 2. 정량적 결과

실험 결과, 비디오 기반 모델이 이미지 기반 모델보다 압도적인 성능을 보였다.

| Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: |
| ImageNet ViT | $52.31\%$ | $88.46\%$ |
| VideoMAE | $68.61\%$ | $92.07\%$ |
| **Ours (MVD)** | $\mathbf{72.93\%}$ | $\mathbf{94.14\%}$ |

MVD 모델은 ImageNet ViT 대비 $\text{Top-1 Accuracy}$ 기준 $20.62\%$p 높은 성능을 보였으며, VideoMAE보다도 $4.32\%$p 더 우수한 결과를 기록하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

본 연구 결과는 수술 단계 인식 작업에서 **시간적 의존성(Temporal dependencies)**을 캡처하는 것이 얼마나 중요한지를 명확히 보여준다. 단일 프레임만으로는 구분이 어려운 수술 단계들이 비디오 클립 내의 움직임과 흐름을 통해 더 정확하게 식별될 수 있음을 시사한다. 특히 MVD가 가장 높은 성능을 낸 것은, 대규모 비디오 데이터(Kinetics-400)와 이미지 모델의 정제된 지식을 동시에 활용하는 distillation 기법이 의료 영상과 같이 특수하고 데이터가 부족한 도메인에서 효과적임을 입증한다.

### 2. 한계 및 논의사항

- **데이터셋 규모:** 사용된 데이터셋이 17개의 비디오로 매우 제한적이다. 이는 딥러닝 모델을 학습시키기에 상당히 적은 양이며, 모델이 특정 환자의 사례에 과적합되었을 가능성을 완전히 배제할 수 없다.
- **실시간성 미검증:** 논문에서는 오프라인 분석 결과만을 제시하였으며, 실제 수술실에서 실시간(Real-time)으로 적용 가능할 정도의 추론 속도(Inference latency)에 대해서는 언급되지 않았다.
- **클래스 불균형:** Fig 2에서 클래스별 빈도 분포가 상이함을 알 수 있는데, 이러한 데이터 불균형(Class Imbalance) 문제를 해결하기 위한 구체적인 손실 함수(Loss function) 수정이나 샘플링 전략에 대한 설명이 부족하다.

## 📌 TL;DR

본 논문은 흉부 외과 수술의 단계 인식을 위해 이미지 기반 모델(ViT)과 비디오 기반 모델(VideoMAE, MVD)의 성능을 비교 분석하였다. 실험 결과, 시공간적 특징과 교사 모델의 지식을 활용하는 **Masked Video Distillation (MVD)** 모델이 $\text{Top-1 Accuracy } 72.93\%$로 가장 우수한 성능을 기록하였다. 이 연구는 향후 수술 워크플로우의 자동 요약, 외과의 숙련도 평가 및 실시간 수술 지원 시스템 구축을 위한 기초 연구로서 중요한 가치를 지닌다.

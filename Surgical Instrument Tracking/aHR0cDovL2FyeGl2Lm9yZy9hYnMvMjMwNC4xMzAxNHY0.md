# Methods and datasets for segmentation of minimally invasive surgical instruments in endoscopic images and videos: A review of the state of the art

Tobias Rueckert, Daniel Rueckert, Christoph Palm (2023)

## 🧩 Problem to Solve

최소 침습 수술(Minimally Invasive Surgery, MIS)은 기존 수술 방식에 비해 환자의 회복 속도를 높이고 외상을 줄이는 장점이 있어 현대 수술의 표준이 되었다. 그러나 좁은 시야와 복잡한 손-눈 협응(hand-eye coordination) 등의 한계가 존재하며, 이를 해결하기 위해 컴퓨터 및 로봇 보조 최소 침습 수술(Robot-Assisted Minimally Invasive Surgery, RAMIS) 연구가 활발히 진행되고 있다.

특히 수술 도구의 위치와 종류를 정확하게 파악하는 것은 수술 기술 평가, 워크플로우 최적화, 주니어 외과의 교육 등을 위해 매우 중요하다. 기존에는 전자기적 방법이나 외부 마커(marker)를 부착하여 도구를 추적했으나, 이는 수술 준비 과정의 복잡성을 증가시키고 기존 워크플로우에 통합하기 어렵다는 단점이 있다. 따라서 마커 없이 순수하게 시각적 정보만을 이용한 수술 도구의 분할(Segmentation) 및 추적(Tracking) 기술이 필요하다.

본 논문의 목표는 2015년부터 2023년까지 발표된 내시경 이미지 및 비디오에서의 수술 도구 분할 및 추적에 관한 최신 연구 동향을 체계적으로 분석하고, 사용된 데이터셋의 특성을 파악하며, 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 도구 분할 분야의 광범위한 문헌 조사를 통해 다음과 같은 분석 체계를 제공한 것이다.

1. **데이터셋의 체계적 분류 및 정량화**: 로봇 및 비로봇 수술 데이터셋을 식별하고, 문헌에서 각 데이터셋이 얼마나 빈번하게 사용되었는지 정량적으로 분석하였다.
2. **분할 방법론의 다각도 분석**: Semantic Segmentation과 Instance Segmentation을 구분하고, 다시 이를 단일 프레임(Single-frame) 방식과 시공간 정보(Spatio-temporal information) 활용 방식으로 세분화하여 분석하였다.
3. **학습 전략 및 아키텍처 트렌드 파악**: 지도 학습(Supervised), 준지도 학습(Semi-supervised), 약지도 학습(Weakly-supervised) 등의 학습 전략과 Attention Mechanism의 적용 여부, 추론 속도(Inference Speed) 등을 심도 있게 고찰하였다.
4. **실제 응용 가능성 검토**: 분할 결과가 실제 로봇 제어(Visual Servo Control) 등에 어떻게 활용되는지 분석하여 기술의 임상적 적용 가능성을 논의하였다.

## 📎 Related Works

저자들은 2015년 이후의 시각 기반 접근 방식에 초점을 맞춘 기존 리뷰 논문들(Bouget et al., Yang et al., Anteby et al., Rivas-Blanco et al. 등)을 검토하였다.

기존 연구들의 한계점은 다음과 같다.

- **범위의 차이**: 많은 기존 리뷰들이 이진 도구 존재 여부 감지(Binary tool presence detection), 수술 단계 인식(Surgical phase recognition), 또는 수술 기술 평가에 집중했으며, 픽셀 단위의 정밀한 '분할(Segmentation)' 연구는 상대적으로 비중이 낮았다.
- **검색 키워드의 한계**: 기존 연구들은 주로 "detection"이라는 용어를 사용하였으며, "segmentation"이라는 키워드를 명시적으로 포함하여 심층 분석한 사례가 부족했다.
- **최신성 및 투명성 부족**: 급격한 기술 발전으로 인해 과거의 리뷰 결과가 현재의 SOTA(State-of-the-Art)를 반영하지 못하며, 일부 논문은 검색 플랫폼, 검색어, 포함/제외 기준 등 방법론적 상세 설명이 부족하여 재현성이 떨어진다.

본 논문은 이러한 한계를 극복하기 위해 "segmentation"에 초점을 맞춘 체계적인 문헌 검색 방법론을 적용하여 분석을 수행하였다.

## 🛠️ Methodology

본 보고서에서 분석한 논문의 리뷰 방법론은 다음과 같다.

### 1. 문헌 검색 및 선택 절차

- **검색 플랫폼**: Google Scholar, Web of Science, PubMed.
- **검색어**: "instrument segmentation", "instrument tracking", "surgical tool segmentation", "surgical tool tracking".
- **대상 기간**: 2015년 1월 ~ 2023년 7월.
- **선택 기준**: 영어로 작성된 피어 리뷰(Peer-review) 논문이어야 하며, 내시경 이미지 데이터를 사용하고, 마커를 사용하지 않는 순수 시각적 방법론을 제시해야 한다. 최종적으로 741개의 논문 중 123개를 선정하여 분석하였다.

### 2. 분석 프레임워크

분석 대상 논문들을 다음과 같은 기준으로 분류하였다.

- **분할 유형 (Segmentation Type)**:
  - **Semantic Segmentation**: 픽셀 단위로 클래스를 구분. (Binary, Instrument Type, Instrument Part로 세분화)
  - **Instance Segmentation**: 동일 클래스 내에서도 개별 객체를 구분.
- **시간적 정보 활용 여부**:
  - **Single Frame**: 단일 이미지 내에서 분할 수행.
  - **Temporal Information**: 비디오 프레임 간의 연속성을 활용하여 추적 및 분할 품질 향상.
- **학습 전략 (Learning Strategy)**:
  - **SV (Supervised)**: 정밀한 ground truth 마스크를 사용한 학습.
  - **SE (Semi-supervised)**: 소량의 라벨링 데이터와 다량의 미라벨링 데이터를 함께 사용.
  - **WE (Weakly-supervised)**: Bounding box나 스케치(Scribble) 같은 거친 라벨을 사용.
  - **SN (Synthetic Data)**: 합성 데이터를 생성하여 학습에 활용.
  - **DA (Domain Adaptation)**: 시뮬레이션 데이터에서 실제 데이터로 도메인 전이 수행.

### 3. 주요 기술적 요소 분석

- **Attention Mechanisms**: CNN의 한계를 극복하기 위해 유용한 특징을 추출하고 무관한 특징을 억제하는 Attention 모듈(예: SE block, Transformer-based attention)의 적용 방식을 분석하였다.
- **추론 속도 (Inference Speed)**: 실시간 적용 가능성을 평가하기 위해 FPS(Frames Per Second) 단위를 기준으로 분석하였다.
- **시간적 정보 처리**: Recurrent layers(RNN), LSTM, Optical Flow(OF) 등을 이용한 도구 추적 및 일관성 유지 방법론을 살펴 보았다.

## 📊 Results

### 1. 데이터셋 분석

분석 결과, EndoVis-2017 데이터셋이 가장 많이 사용되었으며, 그 뒤를 이어 Private Dataset(비공개 데이터셋)의 사용 빈도가 높았다.

- **주요 공개 데이터셋**: EndoVis (2015, 2017, 2018, 2019), Kvasir Instrument, RoboTool, UCL dVRK 등이 있다.
- **특징**: 최근에는 실제 수술 환경의 복잡성을 반영하여 Robustness와 Generalization 능력을 평가하려는 경향이 강해지고 있으며, 라벨링 비용을 줄이기 위해 합성 데이터(Synthetic data)를 생성하는 방식(예: Laparoscopic I2I Translation)이 증가하고 있다.

### 2. Semantic Segmentation 결과

- **주류 방법론**: 대부분의 연구가 지도 학습(Supervised Learning)에 의존하고 있으며, 단일 프레임 분할 연구가 시간적 정보를 활용한 연구보다 훨씬 많다.
- **분할 수준**: 단순한 이진 분할(Binary: 도구 vs 배경)이 가장 많으며, 도구의 종류(Type)나 부분(Part)을 구분하는 연구는 상대적으로 적다.
- **기술 트렌드**: Attention mechanism을 도입하여 segmentation quality를 높이려는 시도가 최근 급증하고 있으며, 특히 Encoder-Decoder 구조의 Skip connection이나 Bottleneck layer에 Attention을 적용하는 방식이 효과적임이 확인되었다.

### 3. Instance Segmentation 결과

- **연구 현황**: Semantic Segmentation에 비해 연구 편수가 현저히 적다. 하지만 동일한 종류의 도구가 여러 개 등장하는 수술 환경에서 개별 도구를 식별하고 추적하는 데 필수적이다.
- **학습 전략**: 분석된 모든 Instance segmentation 연구가 지도 학습(Supervised) 방식을 사용하고 있으며, 합성 데이터의 활용이 거의 없다는 점이 특징이다.

### 4. 시간적 정보 및 추적 (Tracking)

- **방법론**: RNN/LSTM을 통한 특징 집계(Aggregation)나 Optical Flow를 이용한 프레임 간 이동 예측 방식이 주로 사용된다.
- **성능**: 시간적 정보를 통합했을 때 단일 프레임 방식보다 분할의 일관성이 높아지고 추론 속도가 향상되는 경우가 많다.

## 🧠 Insights & Discussion

### 1. 강점 및 긍정적 동향

- **기술적 성숙도**: 딥러닝, 특히 Attention 기반 아키텍처의 도입으로 분할 정밀도가 크게 향상되었으며, 일부 모델은 실시간 처리가 가능한 수준에 도달했다.
- **데이터 효율성**: 합성 데이터 생성 및 Domain Adaptation 기술의 발전으로 수작업 라벨링의 고충을 덜어내려는 노력이 가시화되고 있다.

### 2. 한계 및 비판적 해석

- **데이터 폐쇄성 (Reproducibility Issue)**: 상당수의 연구가 Private Dataset을 사용하여 결과를 도출하고 있다. 이는 결과의 재현성을 불가능하게 하며, 서로 다른 논문 간의 공정한 성능 비교를 어렵게 만든다.
- **분할 수준의 단순함**: 실무적으로는 도구의 종류와 부분(Shaft, Tip 등)을 구분하는 것이 매우 중요함에도 불구하고, 학술적으로는 단순 이진 분할(Binary)에 치중하는 경향이 있다.
- **실시간성 경시**: 많은 논문이 정밀도(Quality) 향상에만 집중하고, 실제 임상 적용의 필수 조건인 추론 속도(Inference Speed)에 대한 분석을 누락하거나 부수적으로 처리하는 경향이 있다.

### 3. 향후 연구 방향

- **Instance-level 연구 확대**: 도구별 고유 ID를 부여하고 추적하는 Instance Segmentation 연구가 더 활발해져야 한다.
- **합성 데이터의 고도화**: 시뮬레이션-실제(Sim-to-Real) 간의 Gap을 줄이는 정교한 합성 데이터 생성 기법이 필요하다.
- **임상 통합**: 단순히 분할 정확도를 높이는 것을 넘어, 이를 로봇 제어 시스템과 결합하여 실제 수술의 효율성을 높이는 End-to-end 시스템 연구가 필요하다.

## 📌 TL;DR

본 논문은 2015~2023년 사이의 내시경 수술 도구 분할 및 추적 연구를 집대성한 리뷰 논문이다. **핵심 요약**은 다음과 같다.

1. **현황**: 딥러닝 기반의 Semantic Segmentation이 주류이며, 최근에는 Attention Mechanism과 합성 데이터를 이용한 Domain Adaptation이 핵심 트렌드이다.
2. **문제점**: 비공개 데이터셋 사용으로 인한 재현성 부족, 단순 이진 분할 중심의 연구, 실시간 처리 속도에 대한 낮은 관심 등이 한계로 지적된다.
3. **전망**: 향후에는 개별 도구를 구분하는 **Instance Segmentation**의 확대와 더불어, 정밀한 **부분 분할(Part segmentation)** 및 실제 **로봇 제어 시스템과의 통합**이 연구의 핵심이 될 것이다.

이 연구는 수술 도구 인식 분야의 연구자들에게 현재 기술의 위치를 알려주고, 단순한 정확도 경쟁에서 벗어나 실제 임상 적용을 위한 실용적 방향(속도, 일반화, 인스턴스 구분)을 제시했다는 점에서 중요한 역할을 한다.

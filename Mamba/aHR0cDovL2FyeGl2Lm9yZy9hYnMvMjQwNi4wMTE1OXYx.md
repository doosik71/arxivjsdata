# Dimba: Transformer-Mamba Diffusion Models

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, Junshi Huang (2024)

## 🧩 Problem to Solve

최근의 Text-to-Image(T2I) 확산 모델(Diffusion Models)은 주로 CNN 기반의 U-Net이나 Transformer 아키텍처를 백본으로 사용하고 있다. 그러나 이러한 구조는 특히 고해상도 이미지 생성이나 긴 컨텍스트를 처리할 때 메모리 사용량과 계산 비용이 급격히 증가하는 치명적인 단점이 있다. 특히 Transformer의 Self-attention 메커니즘은 시퀀스 길이에 따라 계산 복잡도가 이차적으로 증가하는 $\mathcal{O}(N^2)$ 특성을 가지므로, 하드웨어 자원의 제약이 있는 환경에서는 확장이 어렵다.

본 논문의 목표는 Transformer의 강력한 표현 능력과 State Space Model(SSM)인 Mamba의 선형적인 계산 효율성을 결합한 새로운 하이브리드 아키텍처를 제안함으로써, 메모리 사용량을 줄이고 처리량(Throughput)을 높이면서도 기존의 순수 Transformer 기반 모델에 필적하는 이미지 생성 품질을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Transformer 레이어와 Mamba 레이어를 순차적으로 교차 배치한 하이브리드 구조를 설계하는 것이다. 이를 통해 Attention 메커니즘이 제공하는 전역적인 문맥 파악 능력과 Mamba가 제공하는 효율적인 시퀀스 모델링 능력을 동시에 확보하고자 하였다.

주요 기여 사항은 다음과 같다.

1. **하이브리드 아키텍처 제안**: Attention 레이어와 Mamba 레이어를 결합한 Dimba 구조를 통해 메모리 효율성과 생성 성능 사이의 최적의 균형을 찾아내었다.
2. **고품질 데이터셋 구축**: 단순한 웹 크롤링 데이터의 한계를 극복하기 위해, 자동 점수 측정기(Automatic Scorer)와 고급 캡셔닝 모델(ShareCaptioner)을 사용하여 미학적으로 우수하고 설명이 정밀한 대규모 이미지-텍스트 데이터셋을 구축하였다.
3. **단계적 학습 전략**: 대규모 데이터 기반의 사전 학습(Pre-training) 후, 고해상도 적응(Resolution Adaptation) 및 소규모 정밀 데이터셋을 이용한 품질 튜닝(Quality Tuning) 과정을 거치는 단계적 전략을 도입하여 학습 효율을 극대화하였다.

## 📎 Related Works

기존의 T2I 확산 모델들은 U-Net이나 Transformer를 사용하여 노이즈를 예측한다. CNN 기반 U-Net은 다운샘플링과 업샘플링 블록 및 스킵 커넥션을 통해 특징을 추출하며, Transformer 기반 모델은 일부 블록을 Self-attention으로 대체하여 확장성을 높였다. 그러나 두 방식 모두 고해상도 처리 시 계산 비용 문제가 여전히 존재한다.

최근 등장한 State Space Model(SSM), 특히 Mamba는 시퀀스 길이에 대해 선형적인 복잡도를 가지며 하드웨어 최적화 알고리즘을 통해 Transformer의 대안으로 주목받고 있다. 일부 연구에서 Mamba를 시각 데이터 처리나 특정 태스크에 적용하려는 시도가 있었으나, 본 논문은 이를 확산 모델의 백본으로 가져와 Transformer와 혼합하고, 특히 언어 모델의 인과적(Causal) 구조가 아닌 이미지 생성에 적합한 양방향(Bi-directional) 구조로 적용했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 Dimba Block

Dimba는 Transformer 레이어와 Mamba 레이어가 교차로 쌓인 구조를 가진다. 이들의 기본 단위인 'Dimba Block'은 다음과 같이 구성된다.

1. **Attention 및 Mamba 레이어**: Attention 레이어와 Mamba 레이어가 $1:K$ 비율로 배치된다. (실험에서는 $K=1$로 설정하여 동일한 비율로 배치하였다.)
2. **Cross-Attention**: 텍스트 조건부 정보(Conditional Information)를 통합하기 위해 Cross-attention 레이어를 사용한다.
3. **Feed-Forward Network (MLP)**: 각 블록의 마지막에 MLP 레이어를 배치하여 비선형성을 추가한다.
4. **AdaLN (Adaptive Layer Normalization)**: 학습의 안정성을 위해 시간 단계(Time-step) 정보를 투영한 값을 AdaLN 레이어에 삽입한다.

전체 파이프라인은 $\text{T5-XXL}$ 텍스트 인코더를 통해 텍스트 특징을 추출하고, 이를 Dimba Block의 Cross-attention에 입력하며, 이미지의 잠재 공간(Latent Space) 처리를 위해 $\text{SDXL}$의 사전 학습된 VAE를 사용한다.

### 데이터셋 구축 (Dataset Construction)

웹에서 수집한 데이터의 낮은 정렬도와 단순한 캡션 문제를 해결하기 위해 다음과 같은 절차를 거쳤다.

- **품질 필터링**: $\text{LAION-Aesthetics-Predictor V2}$를 사용하여 미학적 점수가 높은 이미지들을 선별하였다.
- **정밀 재캡셔닝**: $\text{ShareCaptioner}$를 이용하여 "이미지를 종합적이고 상세하게 분석하라"는 프롬프트로 고밀도의 캡션을 생성하였다. 그 결과 평균 캡션 길이가 185단어까지 증가하여 묘사력이 크게 향상되었다.
- **데이터 구성**: 약 4,300만 개의 이미지-텍스트 쌍으로 구성된 내부 데이터셋을 구축하였으며, 특히 매우 높은 품질의 이미지 60만 개를 별도로 추출하여 품질 튜닝용으로 사용하였다.

### 학습 전략 (Training Strategy)

- **Quality Tuning**: 전체 데이터로 사전 학습을 진행한 후, 60만 개의 고품질 데이터셋으로 미세 조정(Fine-tuning)을 수행한다. 이때 과적합을 방지하기 위해 Early Stopping을 적용하고 인간의 육안 관찰을 통해 최적의 체크포인트를 선택하였다.
- **Resolution Adaptation**: 저해상도 모델의 위치 임베딩(Position Embedding, PE)을 보간하여 고해상도 모델의 초기값으로 사용하는 $\text{PE Interpolation}$ 기법을 적용하였다. 이는 Mamba의 암시적 위치 모델링 능력과 결합되어 고해상도 학습 시 수렴 속도를 비약적으로 높였다.

## 📊 Results

### 정량적 성능 분석 (Image Quality & Efficiency)

FID-30K 지표를 통해 이미지 품질을 측정한 결과, Dimba-L(0.9B)은 8.93, Dimba-G(1.8B)는 8.15를 기록하였다. 특히 놀라운 점은 학습 효율성이다.

- **학습 시간**: $\text{SDv1.5}$가 6,250 A100 GPU days를 소모한 반면, Dimba-L은 단 704 GPU days만으로 유사한 성능에 도달하였다 (약 11.2% 수준).
- **데이터 양**: $\text{SDv1.5}$가 20억 개의 이미지를 사용한 것에 비해, Dimba는 4,300만 개의 이미지(약 2.0%)만으로 경쟁력 있는 결과를 냈다.

### 이미지-텍스트 정렬 (Compositionality)

$\text{T2I-CompBench}$ 벤치마크에서 Dimba-G는 속성 바인딩(Attribute Binding), 객체 관계(Object Relationship), 복잡한 구성(Complex Composition) 등 모든 지표에서 우수한 성적을 거두었으며, 특히 $\text{SDXL}$나 $\text{PixArt}$와 비교했을 때도 대등하거나 더 높은 성능을 보였다.

### 정성적 평가 및 사용자 연구

- **User Study**: 200개의 프롬프트를 사용하여 인간 평가를 진행한 결과, Dimba-G는 $\text{SDXL}$, $\text{PixArt}$ 등 메인스트림 모델과 비교해 이미지 품질과 프롬프트 준수 능력 면에서 경쟁력이 있음을 확인하였다.
- **AI Preference**: $\text{GPT-4 Vision}$을 평가자로 사용하여 투표를 진행한 결과, 인간 평가와 일관되게 Dimba의 우수성이 입증되었다.

## 🧠 Insights & Discussion

### 강점 및 효율성

본 논문은 아키텍처의 효율적 설계(Transformer-Mamba 하이브리드)와 데이터의 고도화(정밀 캡셔닝 및 미학 필터링)가 결합되었을 때, 막대한 컴퓨팅 자원 없이도 고성능 T2I 모델을 구축할 수 있음을 증명하였다. 특히 $\text{PE Interpolation}$을 통한 고해상도 적응 방식은 하이브리드 모델에서 매우 효율적으로 작동함을 보여주었다.

### 한계 및 비판적 해석

논문에서도 명시했듯이, 학습 데이터의 편향으로 인해 특정 스타일이나 장면 생성에 어려움이 있으며, 특히 확산 모델의 고질적인 문제인 텍스트 생성 오류와 손(Hand) 묘사의 불완전함이 여전히 존재한다. 또한, 매우 복잡한 프롬프트의 경우 완벽하게 정렬되지 않는 경우가 발견되었다.

비판적으로 보았을 때, Dimba의 성능 향상이 순수하게 하이브리드 구조 덕분인지, 아니면 $\text{ShareCaptioner}$를 통한 고품질 데이터셋의 영향이 더 큰지에 대한 정밀한 Ablation Study가 추가될 필요가 있다. 데이터의 질이 비약적으로 상승했기 때문에, 동일한 데이터를 순수 Transformer 모델에 적용했을 때와 비교한 성능 차이가 명확히 제시되어야 하이브리드 구조의 순수한 이점을 완전히 이해할 수 있을 것이다.

## 📌 TL;DR

Dimba는 Transformer의 표현력과 Mamba의 선형적 효율성을 결합한 하이브리드 확산 모델이다. 고도로 정제된 4,300만 개의 이미지-텍스트 데이터셋과 단계적 학습 전략(사전 학습 $\rightarrow$ 해상도 적응 $\rightarrow$ 품질 튜닝)을 통해, 기존 모델 대비 극히 적은 학습 시간과 데이터량만으로도 최신 T2I 모델에 필적하는 고품질 이미지 생성 능력을 구현하였다. 이 연구는 향후 저비용 고효율의 대규모 생성 모델 설계에 중요한 가이드라인을 제공한다.

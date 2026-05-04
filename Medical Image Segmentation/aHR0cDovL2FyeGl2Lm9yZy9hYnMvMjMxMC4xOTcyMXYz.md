# PROMISE: PROMPT-DRIVEN 3D MEDICAL IMAGE SEGMENTATION USING PRETRAINED IMAGE FOUNDATION MODELS

Hao Li, Han Liu, Dewei Hu, Jiacheng Wang, Ipek Oguz (2023)

## 🧩 Problem to Solve

본 연구는 의료 영상 분석에서 고질적으로 발생하는 데이터 획득의 어려움과 전문적인 라벨링 데이터의 부족 문제를 해결하고자 한다. 자연 영상(Natural Image) 도메인에서 학습된 사전 학습 모델(Pretrained Model)을 의료 영상 도메인으로 전이 학습(Transfer Learning)시키는 것은 효율적인 전략이 될 수 있으나, 다음과 같은 세 가지 주요 장벽이 존재한다.

첫째, 자연 영상과 의료 영상 간의 대조도(Contrast) 및 질감(Texture) 특성이 매우 다르다는 점이다. 둘째, 개개인마다 해부학적 구조의 변동성(Anatomical Variability)이 커서 일반화된 세그멘테이션이 어렵다는 점이다. 셋째, 기존의 많은 사전 학습 모델이 2D 기반으로 설계되어 있어, 3D 의료 데이터가 가진 깊이 방향의 공간적 문맥(Depth-related Spatial Context)을 손실하는 문제가 발생한다.

따라서 본 논문의 목표는 사전 학습된 2D 이미지 파운데이션 모델의 지식을 유지하면서, 단일 포인트 프롬프트(Single Point Prompt)만으로도 강건한 3D 의료 영상 세그멘테이션을 수행할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Segment Anything Model (SAM)**의 강력한 이미지 표현 능력을 3D 의료 영상에 적응시키기 위해 **경량 어댑터(Lightweight Adapter)**와 **하이브리드 인코더 구조**를 도입하는 것이다.

주요 기여 사항은 다음과 같다.
1. **3D 적응형 파운데이션 모델 구조**: SAM의 Vision Transformer(ViT)를 기반으로 하되, Depth-wise Convolution을 사용하는 경량 어댑터를 통해 3D 공간 문맥을 캡처함으로써 모델의 가중치를 완전히 업데이트하지 않고도 효율적인 도메인 전이를 달성하였다.
2. **하이브리드 인코더 설계**: Transformer 인코더가 가진 글로벌 정보 추출 능력과 CNN 인코더가 가진 로컬 세부 정보 추출 능력을 결합하여, 특히 경계가 모호한 종양(Tumor) 세그멘테이션의 정확도를 높였다.
3. **Boundary-aware Loss 제안**: 별도의 오프라인 엣지 맵 생성 없이 학습 과정에서 즉시 사용할 수 있는 경계 인식 손실 함수를 제안하여, 불규칙한 모양의 객체에 대해 정밀한 경계 예측이 가능하도록 하였다.

## 📎 Related Works

최근 SAM과 같은 이미지 파운데이션 모델이 등장하며 제로샷(Zero-shot) 성능이 비약적으로 향상되었으나, 의료 분야에 직접 적용하기에는 앞서 언급한 도메인 격차가 크다. 이를 해결하기 위해 모델 전체를 미세 조정(Full Fine-tuning)하는 방식이 시도되었으나, 이는 막대한 계산 자원을 소모한다는 단점이 있다.

이에 대한 대안으로 LoRA(Low-Rank Adaptation)나 경량 어댑터와 같은 파라미터 효율적 미세 조정(Parameter-efficient Fine-tuning) 방식이 연구되었다. 특히 3DSAM-adapter와 같은 최신 연구는 SAM을 3D 의료 영상에 적용하려 했으나, Transformer 블록당 단일 어댑터만을 사용하여 자연 영상과 의료 영상 사이의 큰 차이를 완전히 극복하지 못했으며, 세부적인 디테일을 캡처하는 능력이 부족하여 종양 세그멘테이션에서 최적의 결과를 내지 못하는 한계가 있었다. ProMISe는 이러한 한계를 극복하기 위해 듀얼 어댑터 구조와 CNN 보완 경로를 도입하였다.

## 🛠️ Methodology

### 전체 파이프라인
ProMISe는 3D 입력 이미지와 단일 포인트 프롬프트를 입력으로 받는다. 전체 구조는 **Transformer 인코더**와 **CNN 인코더**라는 두 가지 상호 보완적인 경로로 구성되며, 추출된 특징은 **Prompt 인코더**와 결합되어 최종적으로 **Mask 디코더**를 통해 세그멘테이션 마스크를 생성한다.

### 주요 구성 요소 및 역할
1. **Transformer Encoder**: SAM의 ViT-B 가중치를 기본으로 사용한다. 3D 데이터를 처리하기 위해 학습 가능한 Depth Embedding 레이어를 추가하였다. 특히, 각 Transformer 블록 내에서 입력 단계뿐만 아니라 출력 단계 직전에도 Depth-wise Convolution 기반의 경량 어댑터를 배치하여 3D 공간 정보를 정교하게 반영하고 도메인 전이 효율을 높였다.
2. **CNN Encoder**: Transformer가 놓치기 쉬운 로컬한 세부 정보를 캡처하기 위한 얕은 구조의 CNN 네트워크이다. 이는 특히 종양의 모호한 경계를 명확히 하는 데 기여한다.
3. **Prompt Encoder**: 포인트 프롬프트와 Transformer 인코더에서 추출된 이미지 임베딩을 입력으로 받는다. Visual Sampling을 통해 포인트 임베딩과 이미지 임베딩의 시맨틱 특징을 정렬시킨 후, Self-attention과 Cross-attention 과정을 거쳐 마스크 디코더에 전달한다.
4. **Mask Decoder**: 2D 기반의 기존 디코더를 그대로 사용하는 대신, 3D 특징을 효율적으로 처리할 수 있도록 처음부터 학습된(from scratch) 얕은 CNN 구조의 디코더를 설계하였다.

### 손실 함수 및 학습 절차
본 모델은 구조적 정보와 경계 정보를 동시에 학습하기 위해 하이브리드 손실 함수를 사용한다.

**1. Structural Loss ($L_{structural}$)**:
객체의 전반적인 구조를 잡기 위해 Dice Loss와 Cross-Entropy Loss를 결합하여 사용한다.
$$L_{structural} = L_{Dice} + L_{CE}$$

**2. Boundary-aware Loss ($L_{boundary}$)**:
정밀한 경계 묘사를 위해 제안된 손실 함수이다. 평균 풀링(Average-pooling, 커널 크기 5) 연산 $P_{ave}$를 이용하여 예측 마스크 $S$와 정답 마스크 $G$로부터 부드러운 경계 맵(Smooth Boundary Map)을 생성한다.
$$B(M) = |M - P_{ave}(M)|$$
이후, 생성된 경계 맵 간의 MSE(Mean Squared Error)를 계산하여 세부 윤곽을 복원한다.
$$L_{boundary} = L_{MSE}(B(S), B(G))$$

**최종 목적 함수**:
$$L(S,G) = \lambda_1 L_{structural}(S,G) + \lambda_2 L_{boundary}(B(S), B(G))$$
여기서 $\lambda_1 : \lambda_2 = 1 : 10$의 비율로 설정하여 경계 학습에 더 큰 비중을 두었다.

## 📊 Results

### 실험 설정
- **데이터셋**: Medical Segmentation Decathlon의 대장(Colon) 및 췌장(Pancreas) 종양 데이터셋을 사용하였다.
- **비교 대상**: nnU-Net, 3D UX-Net, nnFormer, Swin-UNETR 및 최신 적응형 모델인 3DSAM-adapter.
- **평가 지표**: Dice score(영역 중첩도)와 Normalized Surface Dice (NSD, 표면 거리 기반 정확도)를 사용하였다.

### 정량적 결과
실험 결과, ProMISe는 단일 포인트 프롬프트만으로도 모든 비교 모델보다 우수한 성능을 보였다.
- **대장 종양**: Dice $66.81\%$, NSD $81.24\%$를 기록하며 3DSAM-adapter($57.32\% / 73.65\%$) 대비 압도적인 성능 향상을 보였다.
- **췌장 종양**: Dice $57.46\%$, NSD $79.76\%$를 기록하여 다른 모델들을 상회하였다.

### 절제 연구 (Ablation Study)
- **어댑터 효과**: 듀얼 어댑터를 사용했을 때 성능이 유의미하게 향상되었다.
- **Boundary Loss 효과**: $L_{boundary}$를 적용했을 때 특히 NSD 지표가 크게 상승하여, 경계 예측 능력이 개선됨을 확인하였다.
- **프롬프트 개수**: 프롬프트를 1개에서 10개로 늘렸을 때 성능이 일부 상승하였으나, 그 차이가 크지 않았다. 이는 단일 클릭만으로도 충분히 강력한 성능을 낼 수 있음을 시사하며, 실무적으로 전문가의 개입을 최소화할 수 있는 장점이 된다.

## 🧠 Insights & Discussion

본 논문은 사전 학습된 2D 모델을 3D 의료 영상으로 확장할 때, 단순한 가중치 전이가 아니라 **데이터의 차원 특성(3D Context)**과 **도메인 간의 시각적 차이**를 메우기 위한 전략적 설계가 필수적임을 보여주었다.

특히, CNN 인코더를 병렬로 배치한 하이브리드 구조와 경계 인식 손실 함수($L_{boundary}$)의 조합은 의료 영상 세그멘테이션의 최대 난제 중 하나인 '모호한 경계(Ambiguous Boundaries)' 문제를 효과적으로 해결하였다. 정성적 결과에서도 ProMISe는 타 모델이 놓치는 영역을 정확히 캡처하였으며, 과소 세그멘테이션(Under-segmentation)과 과다 세그멘테이션(Over-segmentation) 문제를 동시에 완화하였다.

다만, 췌장 세그멘테이션에서 일부 영역이 여전히 과소 세그멘테이션되는 경향이 발견되었으며, 이는 향후 개선 과제로 남는다. 또한, 모델의 효율성을 더 높이기 위해 지식 증류(Knowledge Distillation) 기법을 적용하는 방향이 제시되었다.

## 📌 TL;DR

**ProMISe**는 SAM(Segment Anything Model)의 2D 지식을 3D 의료 영상 세그멘테이션으로 전이하기 위해, **듀얼 경량 어댑터**, **CNN-Transformer 하이브리드 인코더**, 그리고 **Boundary-aware Loss**를 도입한 모델이다. 단일 포인트 프롬프트만으로 대장 및 췌장 종양 세그멘테이션에서 SOTA(State-of-the-art) 성능을 달성하였으며, 특히 모호한 종양 경계를 정밀하게 예측하는 능력이 뛰어나 실제 임상 적용 시 전문가의 수동 라벨링 부담을 획기적으로 줄일 수 있는 가능성을 제시하였다.
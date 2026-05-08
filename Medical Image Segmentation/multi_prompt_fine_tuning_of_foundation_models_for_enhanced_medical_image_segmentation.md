# Multi-Prompt Fine-Tuning of Foundation Models for Enhanced Medical Image Segmentation

Xiangru Li, Yifei Zhang, Liang Zhao (2023)

## 🧩 Problem to Solve

본 논문은 자연 이미지 세그멘테이션에서 혁신적인 성능을 보여준 Foundation Model인 Segment Anything Model (SAM)을 의료 영상 분야에 적용할 때 발생하는 성능 저하 문제를 해결하고자 한다. 의료 영상은 여러 장기와 조직이 복잡하게 얽혀 있고 구조적 경계가 모호하여, SAM과 같은 범용 모델을 그대로 적용했을 때 정밀한 경계 묘사 능력이 부족한 한계가 있다.

특히, 기존의 의료 영상용 SAM 적응 연구인 MedSAM 등은 이미지당 단일 프롬프트(single prompt)만을 사용하는 훈련 프레임워크를 채택하였다. 이는 SAM의 Prompt Encoder가 이미지당 여러 개의 프롬프트를 동시에 처리하여 세그멘테이션의 모호성을 줄일 수 있는 잠재 능력을 충분히 활용하지 못한다는 문제를 야기한다. 따라서 본 연구의 목표는 이미지 내의 여러 관심 영역(ROI)을 동시에 처리하는 다중 프롬프트 파인튜닝(Multi-Prompt Fine-Tuning) 프레임워크를 제안하여 의료 영상 세그멘테이션의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'Co-training'**이라 명명된 다중 프롬프트 활용 전략이다. 의료 영상 데이터베이스에 존재하는 장기-병변 쌍(organ-lesion pairs)과 같은 다중 레이블 특성을 활용하여, 한 이미지 내의 여러 정답 마스크(Ground Truth Masks)로부터 생성된 다중 Bounding Box 프롬프트를 동시에 입력하는 방식이다.

연구진은 특정 장기와 그 내부의 병변이 강한 상관관계를 가진다는 점에 주목하였다. 예를 들어, 장기의 위치 정보는 병변의 위치를 보강하는 힌트가 될 수 있으며, 반대로 병변의 위치 정보는 장기의 경계를 확정 짓는 데 도움을 줄 수 있다. 이러한 상호 보완적인 위치 인코딩(positional encoding)을 모델이 학습하게 함으로써, 단일 프롬프트 방식보다 훨씬 낮은 모호성과 높은 정확도를 달성하고자 하였다.

## 📎 Related Works

본 연구는 SAM의 제로샷(zero-shot) 성능과 이를 의료 영상에 이식하려는 시도들을 기반으로 한다. 특히 MedSAM은 범용 의료 영상 세그멘테이션을 위한 파운데이션 모델로서 유의미한 결과를 냈으나, 훈련 과정에서 이미지당 하나의 프롬프트만을 사용했다는 한계가 있다.

기존의 접근 방식들은 대부분 단일 타겟 세그멘테이션에 집중하였으나, 본 논문은 SAM의 기본 설계가 이미 다중 프롬프트를 처리할 수 있도록 설계되었다는 점에 착안하여 이를 의료 영상의 특수성(장기-병변의 공존)과 결합했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

본 연구의 Co-training 프레임워크는 다음과 같은 단계로 구성된다.

- **모델 초기화**: 사전 학습된 SAM ViT-Base 모델을 기반으로 한다.
- **Image Encoder 동결**: 계산 효율성을 높이기 위해 Image Encoder는 가중치를 고정하고, Image Embedding을 미리 생성하여 사용한다.
- **다중 프롬프트 생성**: 한 이미지 내의 서로 다른 두 가지 정답 마스크(예: 장기 마스크와 병변 마스크)를 사용하여 각각의 Bounding Box를 생성한다.
- **Perturbation(섭동) 적용**: 실제 임상 환경에서 발생할 수 있는 휴먼 에러를 모사하고 모델의 강건성(robustness)을 높이기 위해, Bounding Box의 크기를 일정 범위 내에서 무작위로 변형시키는 섭동을 가한다. 이 범위는 ROI의 타입과 크기에 따라 하이퍼파라미터로 설정된다.
- **Prompt Encoder 및 Mask Decoder**: 변형된 다중 Bounding Box들이 Prompt Encoder를 통해 위치 인코딩으로 변환되며, 이는 Image Embedding과 함께 Mask Decoder로 전달되어 각 프롬프트에 대응하는 세그멘테이션 마스크를 생성한다.

### 2. 학습 목표 및 손실 함수

모델의 학습을 위해 Dice Loss와 Cross-Entropy Loss의 가중치 없는 합(unweighted sum)을 손실 함수로 사용한다.

$$ \text{Loss} = \text{Dice Loss} + \text{Cross-Entropy Loss} $$

각 생성된 마스크는 대응하는 정답 마스크와 개별적으로 손실 값이 계산된다. 특이한 점은, 최적화 단계에서 모든 마스크의 손실을 합산하는 대신, **각 에포크(epoch)에서 계산된 손실 값 중 가장 높은 값(highest loss)만을 사용하여 그래디언트를 계산**한다. 이는 두 가지 주석(annotation) 모두에 대해 적절한 최적화가 이루어지도록 보장하기 위한 전략이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Medical Segmentation Decathlon (MSD)에서 추출한 간(Liver) 및 췌장(Pancreas) CT 스캔 데이터를 사용하였다. 3D 스캔을 z축 기준으로 랜덤 슬라이싱하여 2D 이미지로 변환하였으며, 간 데이터 98건, 췌장 데이터 281건을 확보하였다.
- **데이터 분할**: 훈련 70%, 검증 15%, 테스트 15%로 나누어 진행하였다.
- **비교 대상**: MedSAM(Baseline), 단일 프롬프트로 파인튜닝한 모델(Single Prompt), 그리고 제안하는 Co-train 모델을 비교하였다.
- **평가 지표**: $\text{IoU (Intersection over Union)}$, $\text{DSC (Dice Similarity Coefficient)}$, $\text{NSD (Normalized Surface Distance, threshold 1 pixel)}$를 사용하였다.

### 2. 정량적 결과

실험 결과, Co-train 모델이 모든 지표에서 타 모델을 압도하는 성능을 보였다.

- **병변 세그멘테이션(Lesion Segmentation)**: baseline 및 단일 프롬프트 모델 대비 $\text{IoU}$는 $5\sim31\%$, $\text{DSC}$는 $4\sim23\%$, $\text{NSD}$는 $2\sim48\%$ 향상되었다.
- **장기 세그멘테이션(Organ Segmentation)**: $\text{IoU}$ $3\sim28\%$, $\text{DSC}$ $1\sim19\%$, $\text{NSD}$ $2\sim51\%$의 성능 향상을 보였다.

### 3. 정성적 결과

시각화 결과, Co-train 모델이 생성한 마스크가 정답(Ground Truth)과 가장 유사한 형태를 띠었으며, 다른 모델들에서 나타나는 과적합(overfitting)이나 과소적합(underfitting)으로 인한 경계 모호성 문제가 현저히 적음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상에서 상호 연관된 구조(장기와 병변)를 동시에 학습시키는 것이 단일 구조만 학습시키는 것보다 훨씬 효과적임을 입증하였다. 이는 다중 프롬프트가 제공하는 위치 인코딩의 시너지 효과가 세그멘테이션의 모호성을 제거하는 강력한 가이드 역할을 하기 때문이다. 또한, Bounding Box에 섭동을 가함으로써 실제 환경에서의 가변성에 대비한 강건성을 확보하였다.

다만, 논문에서는 다음과 같은 한계와 논의 사항을 언급하고 있다.

- **임상적 의존성**: 자동화된 세그멘테이션 모델에 대한 과도한 의존은 의료진의 검토 소홀로 이어져 오진이나 잘못된 치료 계획을 세울 위험이 있다.
- **일반화 및 편향**: 학습 데이터에 포함되지 않은 다양한 환자군이나 희귀 케이스에 대한 일반화 성능 문제는 여전히 해결해야 할 과제이며, 데이터 편향으로 인한 의료 불평등 문제가 발생할 수 있다.

## 📌 TL;DR

본 논문은 SAM의 다중 프롬프트 처리 능력을 활용하여 의료 영상의 장기와 병변을 동시에 학습시키는 **Co-training 파인튜닝 프레임워크**를 제안하였다. 다중 Bounding Box 프롬프트를 통해 구조적 모호성을 해결함으로써, 단일 프롬프트 기반의 MedSAM이나 일반 파인튜닝 모델보다 월등히 높은 세그멘테이션 정확도를 달성하였다. 이 연구는 향후 다중 타겟 의료 영상 세그멘테이션 효율성을 높이는 데 중요한 기초가 될 것으로 보인다.

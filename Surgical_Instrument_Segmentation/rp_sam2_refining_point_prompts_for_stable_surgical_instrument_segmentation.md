# RP-SAM2: Refining Point Prompts for Stable Surgical Instrument Segmentation

Nuren Zhaksylyk et al. (2025)

## 🧩 Problem to Solve

본 논문은 백내장 수술(Cataract Surgery) 영상에서 수술 도구의 정확한 세그멘테이션(Segmentation) 문제를 다룬다. 수술 도구 세그멘테이션은 외과의의 숙련도 평가 및 워크플로우 최적화에 필수적이지만, 의료 데이터의 특성상 고품질의 어노테이션 데이터 확보가 매우 어렵고 비용이 많이 든다는 한계가 있다.

최근 SAM2와 같은 Prompt-based 모델들이 유연한 세그멘테이션 능력을 보여주었으나, 본 논문은 이러한 모델들이 Point Prompt의 위치에 매우 민감하게 반응한다는 치명적인 약점을 지적한다. 즉, 동일한 객체 내에서도 클릭하는 지점이 약간만 달라져도 결과 마스크가 크게 변하는 불안정성(Inconsistency)이 발생하며, 이는 실제 의료 현장에서 사용자(어노테이션 작업자)에게 매우 정밀한 포인트 위치 선정을 강요하게 되어 반자동 세그멘테이션의 효율성을 저하시킨다. 따라서 본 연구의 목표는 Point Prompt의 위치 변화에 강건하며 안정적인 세그멘테이션 결과를 제공하는 RP-SAM2 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사용자가 입력한 초기 Point Prompt를 이미지의 문맥(Context)에 맞게 최적의 위치로 이동시키는 **Shift Block**을 도입하는 것이다.

단순히 모델 전체를 미세 조정(Fine-tuning)하는 대신, 이미지 임베딩과 포인트 임베딩 간의 상호작용을 통해 포인트를 정제(Refine)함으로써, 사용자가 객체의 어느 지점을 클릭하더라도 모델이 내부적으로 가장 안정적인 세그멘테이션을 유도하는 지점으로 포인트를 보정하도록 설계하였다. 이를 통해 포인트 위치에 따른 성능 편차를 줄이고 세그멘테이션의 일관성을 확보하였다.

## 📎 Related Works

기존의 SAM 및 SAM2 기반 연구들은 주로 고품질의 프롬프트가 주어졌을 때의 성능을 높이거나, Mask Decoder를 개선하는 방향으로 진행되었다. 일부 연구(Stable-SAM, RoBox-SAM, PP-SAM 등)는 노이즈가 섞인 Bounding Box 프롬프트에 대한 강건성을 높이거나, 디코더의 어텐션을 타겟 영역으로 이동시켜 일관성을 높이려는 시도를 하였다.

그러나 본 논문은 **동일 객체 내에서 포인트의 위치 이동이 세그멘테이션 성능에 미치는 영향**을 분석한 연구가 부족했음을 지적한다. 기존 방식들은 주로 프롬프트의 '개수'나 '박스의 정확도'에 집중했으나, RP-SAM2는 포인트의 '정밀한 위치'에 대한 민감도 문제를 해결함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

RP-SAM2는 기존 SAM2의 구조를 유지하되, Point Prompt Encoder 앞단에 **Shift Block**이라는 경량 모듈(약 12.1M 파라미터)을 추가한 구조이다. 전체 파이프라인은 다음과 같다:

1. 이미지 $I$와 사용자 입력 포인트 $P$가 입력된다.
2. SAM2의 Image Encoder $E$와 Prompt Encoder $\xi$가 각각 임베딩을 생성한다.
3. **Shift Block** $F$가 이미지 임베딩 $i$와 포인트 임베딩 $p$를 입력받아 정제된 포인트 $P'$를 계산한다.
4. 정제된 포인트 $P'$가 SAM2의 Mask Decoder $D$로 전달되어 최종 마스크 $\hat{M}'$를 생성한다.

### Shift Block의 상세 동작

Shift Block은 Point-to-Image Cross-attention 메커니즘을 사용하여 포인트의 오프셋(Offset)을 계산한다.

1. **입력 임베딩**: 학습 가능한 위치 임베딩(Positional Embedding)을 더해 $p^* = p + p_{pos}$와 $i^* = i + i_{pos}$를 생성한다.
2. **Cross-attention**: 포인트 쿼리($Q$)와 이미지 키/밸류($K, V$) 간의 어텐션을 통해 포인트 오프셋 토큰 $\Delta p$를 도출한다.
$$\Delta p = \frac{\exp(Q(p^*) \cdot K(i^*)^T)}{\sum \exp(Q(p^*) \cdot K(i^*)^T)} \cdot V(i^*)$$
3. **포인트 재구성**: 계산된 $\Delta p$를 선형 함수 $\delta$와 $\tanh$ 및 $\sigma$(Sigmoid) 함수를 통과시켜 실제 좌표상의 변화량을 계산하고, 이를 기존 포인트 $P$에 더해 정제된 포인트 $P'$를 얻는다.
$$P' = P + \sigma(s_x, s_y) \cdot \tanh(\delta(\Delta p))$$
여기서 $(s_x, s_y)$는 수술 도구의 비정형적 형태를 반영하기 위해 $x, y$축 방향의 시프트를 독립적으로 조절하는 학습 가능한 토큰이다.

### 학습 절차 및 Compound Loss

모델의 효율성을 위해 Image Encoder, Prompt Encoder, Mask Decoder는 동결(Freeze)하고 Shift Block만 학습시킨다. 학습을 위해 다음과 같은 복합 손실 함수(Compound Loss)를 사용한다.
$$L_{total} = \alpha L_{dice} + \beta L_{dist} + \gamma L_{out}$$

- **Dice Loss ($L_{dice}$)**: 정제된 포인트 $P'$로 생성된 마스크 $\hat{M}'$와 정답 마스크 $M$ 사이의 겹침 정도를 최대화한다.
- **Distance Loss ($L_{dist}$)**: 정제된 포인트 $P'$가 객체 내에서 높은 성능을 냈던 후보 포인트 집합 $G$ 중 가장 가까운 포인트와 거리를 좁히도록 유도한다. 지수 함수를 사용하여 거리가 멀수록 더 큰 패널티를 부여한다.
$$L_{dist}(P', G_i) = \frac{1}{2} \left( \exp(\theta \cdot |x' - x_{g_i}|) + \exp(\theta \cdot |y' - y_{g_i}|) - 2 \right)$$
- **Outside-Object Loss ($L_{out}$)**: 정제된 포인트 $P'$가 객체 영역 밖으로 나가지 않도록 제약하는 Binary Cross Entropy 기반의 손실 함수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cataract1k(내부 분포 데이터), CaDIS(OOD 데이터).
- **비교 모델**: SAM2, HQ-SAM2, MedSAM2, SurgSAM2.
- **평가 지표**: mean Dice Similarity Coefficient (mDSC), mean 95th percentile Hausdorff Distance (mHD95).
- **평가 방법**: 각 객체당 10개의 랜덤한 단일 포인트 프롬프트를 생성하여 평균과 표준편차를 측정함으로써 안정성을 평가하였다.

### 주요 결과

1. **성능 및 안정성 향상**: Cataract1k 테스트셋에서 RP-SAM2는 SAM2 대비 mDSC는 약 2% 상승하였고, mHD95는 21.36% 감소하였다. 특히 mDSC의 표준편차가 $\pm 2.09\%$로 매우 낮게 나타나, 포인트 위치에 관계없이 일관된 성능을 보임을 입증하였다.
2. **취약 클래스 개선**: 투명한 특성 때문에 포인트 위치에 매우 민감했던 Lens Injector(LI) 클래스에서 mDSC가 7.89%나 상승하는 가장 큰 폭의 개선을 보였다.
3. **OOD 일반화 능력**: CaDIS 데이터셋 실험 결과, Zero-shot 상황에서는 SAM2보다 낮았으나, 학습 데이터의 40%만 사용하여 Shift Block을 미세 조정했을 때 SAM2보다 훨씬 높은 mDSC와 안정성을 보였다.
4. **의사 라벨(Pseudo Mask) 생성**: RP-SAM2를 통해 생성한 의사 라벨로 SAM2의 Mask Decoder를 미세 조정(SAM2-FT)했을 때, SAM2가 생성한 라벨을 사용했을 때보다 더 높은 성능 향상이 있었다.

## 🧠 Insights & Discussion

### 강점 및 해석

RP-SAM2의 가장 큰 강점은 **프롬프트의 불안정성을 모델 내부의 보정 메커니즘(Shift Block)으로 해결**했다는 점이다. 정성적 분석 결과, 강한 빛 반사(Light-reflection)가 있는 영역을 클릭했을 때 SAM2는 반사 영역을 객체로 오인하여 세그멘테이션하는 경향이 있으나, RP-SAM2는 문맥 정보를 활용하여 포인트를 적절한 위치로 이동시켜 정확한 경계를 회복하는 모습을 보였다. 이는 모델이 단순한 좌표값이 아니라 이미지의 특징(Feature)을 고려하여 프롬프트를 정제하고 있음을 시사한다.

### 한계 및 향후 과제

학습 데이터가 극도로 적은 구간(0-5%)에서는 일부 포인트가 객체 밖으로 시프트되어 Dice score가 0이 되는 사례가 발생하여 mDSC가 일시적으로 하락하는 현상이 관찰되었다. 이는 포인트 이동의 범위나 제약 조건에 대한 추가적인 연구가 필요함을 의미한다. 또한, 현재는 정지 영상(Image)에 집중하고 있으나, 향후에는 비디오 세그멘테이션에서 프롬프트의 위치를 어떻게 추적하고 업데이트할 것인지가 중요한 연구 방향이 될 것이다.

## 📌 TL;DR

본 논문은 SAM2 기반의 수술 도구 세그멘테이션 시 발생하는 **포인트 프롬프트 위치 민감도 문제**를 해결하기 위해, 입력 포인트를 최적의 위치로 보정하는 **Shift Block**과 이를 학습시키기 위한 **Compound Loss**를 제안하였다. RP-SAM2는 기존 모델들보다 높은 세그멘테이션 정확도(mDSC $\uparrow$, mHD95 $\downarrow$)와 뛰어난 일관성(낮은 분산)을 보였으며, 특히 데이터가 부족한 의료 환경에서 반자동 어노테이션의 부담을 줄이고 모델의 신뢰성을 높이는 데 크게 기여할 것으로 기대된다.

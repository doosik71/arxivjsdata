# A Broader Study of Cross-Domain Few-Shot Learning

Yunhui Guo, Noel C. Codella, Leonid Karlinsky, James V. Codella, John R. Smith, Kate Saenko, Tajana Rosing, and Rogerio Feris (2020)

## 🧩 Problem to Solve

본 논문은 Few-Shot Learning(FSL)에서 발생하는 **Cross-Domain** 시나리오의 문제를 다룬다. 일반적으로 FSL은 메타 학습(meta-learning) 단계의 base classes와 메타 테스트 단계의 novel classes가 동일한 도메인에서 샘플링되었다고 가정한다. 그러나 실제 응용 분야(예: 의료 영상, 위성 영상)에서는 base domain과 target domain 사이에 매우 큰 데이터 분포 차이(domain shift)가 존재하며, 이는 기존 FSL 알고리즘의 성능을 급격히 저하시킨다.

특히, 기존의 Cross-Domain FSL 연구들은 여전히 '자연 이미지(natural images)' 범위 내의 데이터셋만을 사용하여 평가되었기 때문에, 시각적 유사성이 매우 낮은 실제 현실 세계의 이미지 도메인(예: 방사선 사진, 피부과 이미지 등)에서의 일반화 성능을 제대로 측정하지 못했다는 한계가 있다. 따라서 본 논문의 목표는 자연 이미지와 시각적 특성이 판이하게 다른 다양한 도메인을 포함하는 새로운 벤치마크인 **BSCD-FSL (Broader Study of Cross-Domain Few-Shot Learning)**을 제안하고, 이를 통해 기존 메타 학습 및 전이 학습(transfer learning) 방법론들을 엄격하게 평가하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **BSCD-FSL 벤치마크 구축**: 자연 이미지와의 유사성을 결정하는 세 가지 직교 기준(1. Perspective distortion의 존재 여부, 2. Semantic content, 3. Color depth)을 정의하고, 이에 따라 유사도가 단계적으로 낮아지는 4가지 데이터셋(CropDiseases, EuroSAT, ISIC2018, ChestX)을 포함하는 벤치마크를 제안하였다.
2. **광범위한 방법론 평가**: 최신 메타 학습 방법론, 전이 학습 기반의 fine-tuning 변형들, 그리고 Cross-Domain FSL 전용 방법론(FWT 등)의 성능을 체계적으로 비교 분석하였다.
3. **메타 학습의 한계 규명**: 극한의 도메인 차이가 존재하는 상황에서는 최신 메타 학습 방법론들이 단순한 fine-tuning보다 평균 12.8% 낮은 정확도를 보이며, 일부 경우에는 무작위 가중치를 가진 네트워크보다 성능이 낮다는 충격적인 결과를 제시하였다.
4. **도메인 유사성과 성능의 상관관계 입증**: 모든 방법론의 정확도가 제안된 도메인 유사성 지표와 상관관계를 가짐을 확인하여, 벤치마크의 설계 의도가 타당함을 입증하였다.

## 📎 Related Works

### Few-Shot Learning (FSL)

FSL은 적은 수의 예제만으로 새로운 클래스를 인식하는 것을 목표로 한다. 주요 접근법은 다음과 같다.

- **Meta-learning**: MatchingNet, MAML, ProtoNet, RelationNet 등과 같이 '학습하는 법을 학습(learning to learn)'하여 빠른 적응을 꾀하는 방법이다.
- **Generative/Augmentation**: 데이터 증강이나 생성 모델을 통해 부족한 샘플 수를 보완하는 방식이다.
- **Transfer Learning**: 사전 학습된 모델의 특징 추출기를 재사용하고 fine-tuning하는 방식이다.

### Domain Adaptation

소스 도메인과 타겟 도메인의 분포 차이를 줄이는 연구로, 주로 marginal distribution을 정렬하거나 Adversarial-based 접근법을 사용한다. 하지만 기존 연구들은 대부분 학습과 테스트 셋의 클래스가 동일하거나 일부 겹친다는 가정을 전제로 한다.

### Cross-Domain Few-Shot Learning

소스와 타겟 도메인이 다르고 클래스 집합이 완전히 분리된(disjoint) 경우를 다룬다. 기존 연구들은 자연 이미지 범위 내에서의 Cross-Domain 설정만을 다루었으며, 의료나 항공 영상과 같은 비정형 도메인으로의 확장 연구는 부재했다.

## 🛠️ Methodology

### 1. 문제 정의 (Problem Formulation)

Cross-Domain FSL은 소스 도메인 $(X_s, Y_s)$와 타겟 도메인 $(X_t, Y_t)$가 존재하며, 두 도메인의 결합 분포 $P_s$와 $P_t$가 서로 다르고($P_{X_s} \neq P_{X_t}$), 레이블 집합 $Y_s$와 $Y_t$가 서로소(disjoint)인 상태에서 정의된다.
모델 $f_\theta$는 소스 도메인에서 학습(또는 메타 학습)된 후, 타겟 도메인의 $K$개 클래스에 대해 각각 $N$개의 샘플을 가진 서포트 셋 $S = \{x_i, y_i\}_{i=1}^{K \times N}$을 통해 적응(adapt)하여 쿼리 셋의 성능을 평가받는다.

### 2. 평가 대상 방법론

- **Meta-learning based**: MatchingNet, MAML, ProtoNet, RelationNet, MetaOpt 등을 평가하였다. 특히 Cross-Domain 전용 기법인 **Feature-wise Transform (FWT)**을 각 메타 학습 모델에 결합하여 성능 변화를 측정하였다.
- **Transfer learning based**:
  - **Single Model Fine-tuning**: Fixed feature extractor, Fine-tuning all layers, Fine-tuning last-k layers, Transductive fine-tuning 등을 비교하였다.
  - **Classifier variants**: 단순 Linear classifier 외에 Mean-centroid classifier와 Cosine-similarity based classifier를 적용하였다.
- **Incremental Multi-model Selection (IMS-f)**: 여러 개의 서로 다른 사전 학습된 모델 $\{M_c\}_{c=1}^C$로부터 최적의 레이어 부분 집합 $I$를 선택하여 특징 벡터를 결합하는 앙상블 방식이다.

최적의 레이어 집합 $I$를 찾는 목적 함수는 다음과 같다.
$$\arg \min_{I \subseteq F} \mathbb{E}_{(x,y) \sim P_t} [\ell(f_s(T(\{l(x) : l \in I\})), y)]$$
여기서 $T()$는 선택된 레이어들의 특징 벡터를 연결(concatenate)하는 함수이며, $f_s$는 선형 분류기이다. 실제로는 계산 복잡도로 인해 2단계 탐욕적 선택(greedy selection) 알고리즘을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CropDiseases(농업), EuroSAT(위성), ISIC2018(피부과), ChestX(흉부 X-ray).
- **기존 모델 학습**: miniImageNet을 base classes로 사용하여 사전 학습 및 메타 학습 수행.
- **평가 지표**: 5-way 5-shot, 20-shot, 50-shot 설정에서 평균 정확도 및 95% 신뢰 구간 측정.
- **백본 네트워크**: ResNet-10 사용.

### 주요 결과

1. **메타 학습 vs 전이 학습**: 모든 데이터셋과 shot 설정에서 전이 학습 기반의 fine-tuning이 메타 학습 방법론들을 압도하였다. 메타 학습 방법론들은 단순 fine-tuning 대비 평균 12.8% 낮은 정확도를 보였다.
2. **메타 학습의 성능 역전**: 최신 SOTA 방법론인 MetaOpt-Net이 ProtoNet과 같은 초기 방법론보다 성능이 낮게 나타났으며, MatchingNet의 경우 무작위 가중치(Random) 네트워크보다 낮은 성능을 기록하기도 하였다.
3. **도메인 유사성과의 관계**: 성능은 $\text{CropDiseases} > \text{EuroSAT} > \text{ISIC} > \text{ChestX}$ 순으로 나타났다. 이는 자연 이미지와의 시각적 유사도가 낮을수록 모든 방법론의 성능이 하락함을 의미하며, 벤치마크의 변별력을 입증한다.
4. **IMS-f의 효용성**: 여러 사전 학습 모델을 결합한 IMS-f 방식이 단일 모델 fine-tuning보다 성능이 높게 나타났으며, 특히 타겟 도메인이 자연 이미지와 매우 다를수록(ChestX, ISIC) 다중 모델의 효과가 크게 나타났다.

## 🧠 Insights & Discussion

### 메타 학습의 실패 원인

본 논문은 메타 학습 방법론들이 base class 데이터의 **태스크 분포(task distribution)에 과적합(overfitting)**되었기 때문이라고 분석한다. 즉, 소스 도메인 내에서 빠르게 적응하는 법은 배웠으나, 소스 도메인과 완전히 다른 분포를 가진 타겟 도메인으로의 일반화 능력은 오히려 저해된 것이다.

### 레이어별 전이 특성

사전 학습된 네트워크의 레이어별 파라미터 변화량을 분석한 결과, 모든 데이터셋에서 **첫 번째 레이어의 변화가 가장 컸다**. 이는 타겟 도메인이 소스 도메인과 다를 경우, 네트워크의 하위 레이어(low-level feature extractor)부터 재조정이 필요함을 시사한다.

### 비판적 해석

- **FWT의 무용성**: 기존 Cross-Domain FSL 연구에서 효과적이라고 알려진 FWT가 본 벤치마크에서는 오히려 성능을 저하시키거나 효과가 없었다. 이는 기존 CD-FSL 연구들이 다룬 도메인 간의 차이가 실제로는 매우 작았음을 방증한다.
- **하이퍼파라미터 튜닝 문제**: 저자들은 소스 도메인과 타겟 도메인이 너무 다르기 때문에 소스 데이터로 튜닝한 하이퍼파라미터가 타겟 도메인에 적절하지 않을 수 있음을 언급하며, 이를 향후 연구 과제로 남겼다.

## 📌 TL;DR

본 논문은 자연 이미지, 위성, 의료 영상 등 시각적 특성이 매우 다른 도메인들을 포괄하는 **BSCD-FSL 벤치마크**를 제안하였다. 실험 결과, 기존의 최신 메타 학습 방법론들이 극한의 도메인 차이 상황에서는 단순한 fine-tuning이나 심지어 무작위 가중치 모델보다 성능이 낮다는 것을 밝혀냈다. 이는 현재의 FSL 연구가 도메인 간의 실제적인 차이를 간과하고 있음을 시사하며, 향후 연구가 더 넓은 범위의 도메인 일반화 능력을 갖추는 방향으로 나아가야 함을 강조한다.

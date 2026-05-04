# Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation

Xixi Jiang, Dong Zhang, Xiang Li, Kangyi Liu, Kwang-Ting Cheng, Xin Yang (2024)

## 🧩 Problem to Solve

본 논문은 **부분 지도 다기관 의료 영상 분할(Partially-Supervised Multi-Organ Medical Image Segmentation, Mo-MedISeg)** 문제를 해결하고자 한다. 이 작업의 목표는 각 데이터셋이 단 하나의 장기 클래스에 대해서만 라벨을 제공하는 여러 개의 부분 라벨링된 데이터셋을 활용하여, 통합된 시맨틱 분할 모델을 개발하는 것이다.

이 문제의 중요성은 의료 영상 분야에서 모든 장기에 대해 정밀한 픽셀 단위 라벨을 확보하는 것이 매우 노동 집약적이고 시간 소모적이며, 전문 지식이 필요하기 때문에 발생한다. 따라서 완전 지도 학습(Fully-supervised learning) 데이터셋을 구축하는 것은 매우 어렵다.

논문에서 지적하는 핵심 문제는 **라벨링된 픽셀과 라벨링되지 않은 픽셀 간의 분포 불일치(Distribution Mismatch)**이다. 구체적으로 두 가지 원인이 존재한다:
1. **과적합(Overfitting):** 라벨링된 전경(Foreground) 픽셀의 수가 제한적이어서 모델이 편향된 분포를 학습하게 된다.
2. **모호한 경계:** 라벨링되지 않은 전경 픽셀과 배경(Background) 픽셀을 구분할 감독 신호가 부족하여, 두 클래스의 특징 공간(Feature manifold)이 서로 얽히게 된다. 특히 췌장(Pancreas)과 같이 배경과의 대비가 낮고 경계가 불분명한 장기에서 이 문제가 심각하며, 이는 결국 부정확한 의사 라벨(Pseudo-label) 생성으로 이어진다.

## ✨ Key Contributions

본 논문은 분포 불일치 문제를 해결하기 위해 **Labeled-to-Unlabeled Distribution Alignment (LTUDA)** 프레임워크를 제안한다. 핵심 아이디어는 다음 두 가지 전략을 통해 라벨링된 픽셀과 라벨링되지 않은 픽셀의 특징 분포를 정렬하는 것이다.

1. **Cross-set Data Augmentation (CDA):** 라벨링된 픽셀과 라벨링되지 않은 픽셀 간의 영역 수준 믹싱(Region-level mixing)을 수행하여 학습 세트를 확장하고 분포의 간극을 메운다.
2. **Prototype-based Distribution Alignment (PDA):** 프로토타입(Prototype) 기반의 분류기를 도입하여 클래스 내 분산은 줄이고 클래스 간 분리도는 높임으로써, 전경과 배경 간의 판별 능력을 강화한다.

## 📎 Related Works

### 관련 연구 및 한계
- **조건 기반 방법(Condition-based methods):** 각 분할 작업을 작업 인식 사전 정보(Task-aware prior)로 인코딩하여 단일 네트워크를 학습시키지만, 다기관 결과를 얻기 위해 여러 번의 추론(Inference)을 수행해야 하므로 시간이 많이 걸린다.
- **의사 라벨링 방법(Pseudo-labeling methods):** 단일 기관 모델들을 사전 학습시켜 의사 라벨을 생성하고 이를 통해 학생 모델을 학습시킨다. 하지만 라벨링된 픽셀과 라벨링되지 않은 픽셀이 동일한 분포를 가진다는 가정을 전제로 하므로, 분포 불일치가 발생할 경우 확인 편향(Confirmation bias)으로 인해 성능이 저하된다.
- **부분 지도 학습의 일반적 접근:** 라벨링되지 않은 클래스를 무시하거나 배경으로 처리하는 방식이 있으나, 전자는 과적합을 유발하고 후자는 거짓 음성(False negative) 오류를 유발하여 모델을 혼란스럽게 만든다.

### 차별점
LTUDA는 기존의 의사 라벨링 방법들이 간과했던 **분포 불일치 문제**를 정면으로 다룬다. 단순히 의사 라벨을 생성하는 것에 그치지 않고, 데이터 증강과 프로토타입 정렬을 통해 특징 분포 자체를 교정함으로써 편향되지 않은 의사 라벨을 생성하도록 유도한다.

## 🛠️ Methodology

### 전체 파이프라인
본 프레임워크는 Teacher-Student 구조를 기반으로 한다. Teacher 모델은 Student 모델의 지수 이동 평균(EMA)을 통해 업데이트된다. 
- **Teacher 모델:** 약한 증강(Weak augmentation)이 적용된 이미지 $X^w$를 입력받아 의사 라벨 $\hat{Y}$를 생성한다.
- **Student 모델:** 강한 증강(Strong augmentation)이 적용된 이미지 $X^s$를 입력받으며, 세 가지 분류기(Linear, Labeled Prototype, Unlabeled Prototype)를 통해 예측을 수행하고 이를 의사 라벨로 감독한다.

### 주요 구성 요소 및 방법론

#### 1. Linear Threshold-based Classifier
모델은 각 장기 클래스에 대해 전경 맵(Foreground maps) $p=\{p_c\}_{c=1}^C$를 출력한다. 최종 클래스 $\hat{y}$는 다음과 같이 결정된다.
$$\hat{y}=
\begin{cases} 
\arg \max_{c\in\{1,\dots,C\}} p_c, & \text{if } \max_{c\in\{1,\dots,C\}} p_c \geq \tau \\ 
0 (\text{background class}), & \text{otherwise}
\end{cases}$$
여기서 $\tau$는 배경과 전경을 구분하는 임계값이다.

#### 2. Cross-set Data Augmentation (CDA)
라벨링된 픽셀과 라벨링되지 않은 픽셀의 분포 간극을 줄이기 위해 CutMix를 사용하여 강한 증강 이미지 $x_s$와 라벨 $\hat{y}_s$를 생성한다.
$$x_s = (1-M) \odot x_a^w + M \odot x_b^w$$
$$\hat{y}_s = (1-M) \odot \hat{y}_a + M \odot \hat{y}_b$$
여기서 $M$은 이진 마스크이며, $x_a^w$와 $x_b^w$는 서로 다른 데이터셋에서 샘플링된 이미지이다. 이를 통해 라벨링된 픽셀의 분포가 라벨링되지 않은 픽셀의 분포에 근사하도록 유도한다.

#### 3. Prototype-based Distribution Alignment (PDA)
특징 공간에서 각 클래스의 평균 특징인 프로토타입을 계산하여 분류를 수행한다.
- **Labeled Prototype Classifier:** 실제 라벨 $Y^l_s$를 사용하여 프로토타입 $P^{l\text{proto}}$를 생성한다. (배경 클래스는 의사 라벨 사용)
- **Unlabeled Prototype Classifier:** 의사 라벨 $\hat{Y}^{ul}_s$를 사용하여 프로토타입 $P^{ul\text{proto}}$를 생성한다.

이 두 분류기와 선형 분류기 간의 일관성을 강제함으로써 특징 분포를 정렬한다. 프로토타입 분류기의 손실 함수 $L_{\text{proto}}$는 다음과 같다.
$$L_{\text{proto}}(X^s, \hat{Y}^s) = L_{CE} + \lambda_1 L_{PPD} + \lambda_2 L_{PPC}$$
- $L_{CE}$: 클래스 간 분류 정확도를 높이는 교차 엔트로피 손실이다.
- $L_{PPD}$ (Pixel-to-Prototype Distance): 픽셀 임베딩과 해당 프로토타입 간의 거리를 직접 최소화하여 클래스 내 분산을 줄인다.
- $L_{PPC}$ (Pixel-to-Prototype Contrastive): 픽셀이 자신의 긍정 프로토타입과는 가깝게, 다른 부정 프로토타입과는 멀게 위치하도록 강제하여 클래스 간 분리도를 높인다.

### 학습 절차
1. **Stage 1:** 선형 분류기만을 사용하여 기본 모델을 학습시킨다 (CDA 적용).
2. **Stage 2:** Stage 1의 가중치로 초기화한 후, PDA 모듈을 추가하여 선형 분류기와 두 프로토타입 분류기를 동시에 학습시키며 분포 정렬을 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋:** 
    - Toy dataset: AbdomenCT-1K에서 샘플링된 50케이스.
    - LSPL dataset: LiTS(간), MSD-Spleen(비장), KiTS(신장), NIH82(췌장)의 합집합.
    - 일반화 테스트: BTCV, AbdomenCT-1K.
- **지표:** Dice Similarity Coefficient (Dice $\uparrow$), Hausdorff Distance (HD $\downarrow$).
- **비교 대상:** Multi-Nets, Med3d, DoDNet, PIPO-FAN, Co-training, Ms-kd, CPS, DMPLS 등.

### 주요 결과
- **Toy Dataset:** 평균 Dice 92.94%를 달성하여 모든 부분 지도 학습 방법론을 압도했으며, 심지어 완전 지도 학습(Fully-supervised) 성능(92.08%)보다 높은 결과를 보였다.
- **LSPL Dataset:** 평균 Dice 91.08%, HD 10.63을 기록하여 SOTA 성능을 달성했다. 특히 경계가 모호한 췌장(Pancreas)과 샘플 수가 적은 비장(Spleen)에서 뚜렷한 성능 향상이 나타났다.
- **일반화 성능:** 외부 데이터셋인 BTCV와 AbdomenCT-1K에서도 타 방법론 대비 우수한 성능을 보이며 강력한 일반화 능력을 입증했다.
- **Ablation Study:** CDA만 적용했을 때보다 PDA를 함께 적용했을 때 Dice 성능이 더욱 향상됨을 확인했으며, CutMix와 같은 Cross-set 증강 전략이 Intra-set 증강(Color jitter, Cutout)보다 훨씬 효과적임을 보였다.

## 🧠 Insights & Discussion

### 강점
- 본 연구는 부분 지도 학습에서 발생하는 성능 저하의 근본 원인이 '분포 불일치'에 있음을 정확히 짚어냈으며, 이를 해결하기 위한 구체적인 메커니즘(CDA, PDA)을 제시했다.
- 특히 프로토타입 기반의 정렬 방식은 추가적인 파라미터 증가 없이 학습 과정에서의 정규화만으로 특징 공간을 최적화하므로 효율적이다.
- 소규모의 부분 라벨링된 데이터만으로도 완전 지도 학습 이상의 성능을 낼 수 있음을 보여줌으로써, 실제 임상 환경에서의 데이터 구축 비용을 획기적으로 줄일 수 있는 가능성을 제시했다.

### 한계 및 가정
- 이미지의 모달리티가 복부 CT 스캔으로 한정되어 있으며, 장기의 위치나 자세가 일관되게 유지된다는 가정을 전제로 한다.
- 프로토타입의 초기 품질이 성능에 영향을 미칠 수 있어 2단계 학습 전략을 사용하는데, 이는 학습 시간을 증가시키는 요인이 된다.

### 비판적 해석
논문은 완전 지도 학습보다 성능이 높게 나온 이유를 CDA를 통한 데이터 확장과 PDA를 통한 클래스 응집력 향상으로 설명한다. 하지만 이는 일종의 강력한 정규화 효과가 작용한 것으로 보이며, 모든 데이터셋에서 동일한 경향이 나타날지는 추가 검증이 필요하다. 또한, 프로토타입 개수($K=5$)에 대한 실험이 포함되어 있으나, 최적의 $K$값이 데이터셋의 복잡도에 따라 어떻게 변하는지에 대한 분석은 부족하다.

## 📌 TL;DR

본 논문은 부분 지도 다기관 의료 영상 분할에서 라벨링된 픽셀과 라벨링되지 않은 픽셀 간의 **분포 불일치(Distribution Mismatch)** 문제를 해결하기 위해 **LTUDA** 프레임워크를 제안한다. **Cross-set Data Augmentation(CDA)**을 통해 데이터 분포의 간극을 메우고, **Prototype-based Distribution Alignment(PDA)**를 통해 특징 공간의 응집도와 분리도를 높임으로써, 매우 적은 양의 라벨만으로도 SOTA 성능(심지어 완전 지도 학습 능가)을 달성하였다. 이 연구는 향후 의료 영상 파운데이션 모델 구축 시 공공 데이터셋의 불완전한 라벨을 효율적으로 활용하는 핵심 기술로 적용될 가능성이 높다.
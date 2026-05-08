# Rethinking Few-Shot Medical Image Segmentation by SAM2: A Training-Free Framework with Augmentative Prompting and Dynamic Matching

Haiyue Zu, Jun Ge, Heting Xiao, Jile Xie, Zhangzhe Zhou, Yifan Meng, Jiayi Ni, Junjie Niu, Linlin Zhang, Li Ni, and Huilin Yang (2025)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 대규모 라벨링 데이터에 대한 높은 의존성이다. 딥러닝 기반의 의료 영상 분석은 정확도가 높지만, 전문 의료진이 직접 라벨링한 대량의 데이터셋이 필요하며 이는 막대한 시간과 비용을 초래한다.

Few-Shot Learning(FSL)이 이 문제의 대안으로 제시되었으나, 기존의 Few-Shot Medical Image Segmentation(FSMIS) 방법론들은 여전히 상당한 양의 데이터셋을 이용한 메타 학습(Meta-training) 단계를 필요로 한다는 한계가 있다. 또한, 기존의 3D 의료 영상 처리 방식은 대부분 영상을 슬라이스 단위(Slice-by-slice)로 처리하는 패러다임에 머물러 있어, 3D 볼륨 데이터가 가진 전체적인 맥락을 충분히 활용하지 못하는 문제가 존재한다.

따라서 본 논문의 목표는 모델의 재학습이나 파라미터 업데이트 없이, 단 하나의 라벨링된 지원 이미지(Support Image)만으로 3D 의료 영상의 타겟 영역을 정확하게 분할할 수 있는 Training-free 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 3D 의료 영상 볼륨을 하나의 비디오 시퀀스로 간주하고, 최신 비전 파운데이션 모델인 SAM2(Segment Anything Model 2)의 비디오 분할 능력을 활용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **데이터 관점의 전환**: 3D 의료 영상을 개별 슬라이스의 집합이 아닌, 연속적인 비디오 시퀀스로 재정의함으로써 SAM2의 시공간적 전파(Propagation) 능력을 의료 영상 분할에 도입하였다.
2. **Augmentative Prompting 및 Dynamic Matching 전략**: 단일 지원 이미지에 광범위한 데이터 증강(Data Augmentation)을 적용하여 다양한 변형을 생성하고, 쿼리 영상의 각 슬라이스와 가장 유사한 증강 이미지를 알고리즘적으로 선택하여 마스크 프롬프트로 사용하는 전략을 제안하였다.
3. **Training-Free 프레임워크**: SAM2의 가중치를 전혀 수정하지 않고 프롬프트 엔지니어링만으로 SOTA(State-of-the-art) 성능을 달성함으로써, 학습 비용을 완전히 제거한 플러그 앤 플레이(Plug-and-play) 방식의 솔루션을 제시하였다.

## 📎 Related Works

### Few-Shot Segmentation (FSS)

초기 FSS 연구들은 지원 브랜치(Support branch)와 쿼리 브랜치(Query branch) 간의 상호작용을 통해 전이 가능한 지식을 학습하는 방식에 집중하였다. 특히 Prototypical Networks를 기반으로 클래스별 프로토타입을 생성하여 매칭하는 방식이 널리 사용되었으며, 이후 Transformer 기반 모델이나 조건부 네트워크(Conditional networks)로 발전하였다.

### Few-Shot Medical Image Segmentation (FSMIS)

의료 영상은 일반 영상에 비해 도메인 간 이질성이 크기 때문에, 이를 해결하기 위한 적응형 로컬 프로토타입 생성이나 자기지도 학습(Self-supervised learning) 기반의 방법론들이 제안되었다. 하지만 이러한 방식들은 여전히 기반 클래스(Base classes)에 대한 광범위한 메타 학습이 선행되어야 한다는 한계가 있다.

### SAM for Medical Images

SAM(Segment Anything Model)은 강력한 일반화 성능을 보이지만, 의료 영상 도메인과의 차이로 인해 직접 적용 시 성능이 저하되는 경향이 있다. 이를 해결하기 위해 의료 전문가의 인터랙티브한 프롬프트 입력이나, 의료 데이터셋을 이용한 미세 조정(Fine-tuning) 방식이 사용되었으나, 이는 각각 확장성 부족과 데이터 의존성 증가라는 문제를 야기한다. 본 논문은 SAM2의 비디오 분할 기능을 활용하여 이러한 한계를 극복하고자 한다.

## 🛠️ Methodology

본 논문이 제안하는 프레임워크는 **지원 세트 구축 $\rightarrow$ 지원-쿼리 매칭 $\rightarrow$ SAM2 기반 프롬프트 구동 분할**의 3단계 파이프라인으로 구성된다.

### 1. Support Set Construction (지원 세트 구축)

단일 지원 이미지-마스크 쌍 $(I_s, M_s)$로부터 다양한 변형을 가진 증강 세트를 생성하여 매칭의 강건성을 높인다.

- **Affine Transformation**: 회전, 크기 조절, 전단, 이동을 포함하는 $3 \times 3$ 행렬 $T_a$를 사용하여 이미지와 마스크를 동시에 변형한다.
$$ T_a = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ 0 & 0 & 1 \end{bmatrix} $$
- **Color Jittering**: 이미지의 밝기, 대비, 채도, 색조를 조정하는 변환 $T_c$를 적용한다. 이때 색상 변환은 이미지 $I_s^a$에만 적용되며 마스크 $M_s^a$는 유지된다.
결과적으로 $N_T \times N_q + 1$ 개의 증강된 이미지-마스크 쌍으로 구성된 확장 지원 세트 $S'$가 구축된다.

### 2. Support-Query Matching (지원-쿼리 매칭)

쿼리 볼륨의 각 슬라이스 $I^q_j$에 대해, 확장 지원 세트 $S'$ 내에서 시각적으로 가장 유사한 지원 이미지 $I'_{s_{i^*(j)}}$를 선택한다. 이때 단순 픽셀 비교가 아닌, 인간의 지각적 유사도와 밀접한 **LPIPS(Learned Perceptual Image Patch Similarity)** 지표를 사용한다. LPIPS는 사전 학습된 CNN(AlexNet 등)의 특징 맵 간 MSE(Mean Squared Error)의 가중치 합으로 계산된다.
$$ L_{LPIPS}(I_1, I_2) = \sum_l w_l \cdot MSE(\Phi^l(I_1), \Phi^l(I_2)) $$
최적의 인덱스 $i^*$는 다음과 같이 결정된다.
$$ i^*(j) = \arg \min_{i \in \{0, 1, \dots, |S|\}} L_{LPIPS}(I^q_j, I^s_i) $$

### 3. Prompt-Driven Segmentation with SAM2

SAM2의 비디오 분할 기능을 활용하여 3D 볼륨을 처리한다.

- **Pseudo Video 구성**: 각 쿼리 슬라이스 $I^q_j$에 대해, 앞서 선택된 최적의 지원 이미지 $I'_{s_{i^*(j)}}$와 쿼리 슬라이스를 순차적으로 배치한 2프레임 시퀀스 $I_j = \langle I'_{s_{i^*(j)}}, I^q_j \rangle$를 구성한다.
- **Mask Prompt 적용**: 선택된 지원 이미지의 마스크 $M'_{s_{i^*(j)}}$를 SAM2의 마스크 프롬프트로 입력한다.
- **분할 수행**: SAM2는 첫 번째 프레임(지원 이미지)의 마스크 정보를 두 번째 프레임(쿼리 슬라이스)으로 전파하여 최종 예측 마스크 $\hat{M}^q_j$를 생성한다.
$$ \hat{M}^q_j = SAM2(I_j, M'_{s_{i^*(j)}}) $$
이 모든 과정은 SAM2의 파라미터 업데이트 없이 수행된다.

## 📊 Results

### 실험 설정

- **데이터셋**: Synapse-CT (복부 CT), CHAOS-MRI (복부 MRI), CMR (심장 MRI) 3종을 사용하였다.
- **평가 지표**: 분할 성능 측정의 표준인 Dice Score(DSC)를 사용하였으며, 5-fold 교차 검증을 통해 평균값을 산출하였다.
- **설정**: 1-way 1-shot 설정에서 평가하였으며, 모델의 성능이 백본의 크기에만 의존하는 것을 방지하기 위해 SAM2의 'tiny' 변형 모델을 사용하였다.

### 정량적 결과

제안 방법은 모든 데이터셋에서 기존 SOTA 방법론들을 상회하는 성능을 보였다.

- **Synapse-CT**: 평균 Dice Score $80.02\%$를 달성하여 기존 SOTA(GMRD) 대비 $1.50\%$ 향상되었으며, 특히 비장(Spleen)과 간(Liver) 영역에서 각각 $5.42\%$, $3.52\%$의 큰 폭의 상승을 보였다.
- **CHAOS-MRI**: 평균 Dice Score $83.30\%$를 기록하여 SOTA 대비 $0.40\%$ 향상되었으며, 특히 신장(Kidney) 분할에서 우수한 성능을 보였다.
- **CMR**: 평균 Dice Score $84.50\%$를 달성하며 SOTA 대비 $5.39\%$라는 압도적인 성능 향상을 기록하였다.

### 분석 및 ablation study

- **모델 크기 영향**: SAM2의 모델 크기가 'tiny' $\rightarrow$ 'small' $\rightarrow$ 'base' $\rightarrow$ 'large'로 증가함에 따라 성능이 일관되게 향상되는 확장성(Scalability)을 확인하였다.
- **증강 횟수($N_T$) 영향**: 증강을 하지 않은 경우($N_T=0$)보다 $N_T=2$ 또는 $4$일 때 성능이 크게 향상되었으며, 이는 지원 세트의 다양성이 매칭 정확도를 높임을 시사한다.
- **유사도 지표 비교**: LPIPS가 SSIM이나 PSNR보다 높은 성능을 보였으며, 이는 지각적 유사도가 의료 영상의 형태적 특징을 포착하는 데 더 효과적임을 입증한다.

## 🧠 Insights & Discussion

본 연구는 3D 의료 영상을 비디오 시퀀스로 재해석함으로써, 복잡한 재학습 과정 없이 파운데이션 모델의 능력을 극대화할 수 있음을 보여주었다. 특히 LPIPS 기반의 동적 매칭과 데이터 증강을 결합한 전략은 단 하나의 지원 이미지만으로도 다양한 해부학적 변이를 극복할 수 있게 하였다.

**강점 및 의의**:
가장 큰 강점은 'Training-free'라는 점이다. 기존 FSMIS 모델들이 겪고 있던 메타 학습의 데이터 의존성과 계산 비용 문제를 완전히 해결하였으며, SAM2라는 강력한 모델을 그대로 활용하는 플러그 앤 플레이 방식이기에 실제 임상 환경에서 빠르게 배포될 가능성이 높다.

**한계 및 논의 사항**:
논문에서 명시적으로 다루지 않았으나, 쿼리 볼륨의 크기가 매우 클 경우 모든 슬라이스에 대해 LPIPS 매칭을 수행하는 과정에서 계산 시간이 증가할 수 있다. 또한, SAM2의 'tiny' 모델로도 SOTA를 달성했지만, 더 큰 모델을 사용할 때의 메모리 요구량과 실시간 처리 속도 간의 트레이드-오프에 대한 분석이 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 3D 의료 영상을 비디오 시퀀스로 간주하고 SAM2의 비디오 분할 능력을 활용한 **Training-free Few-Shot 의료 영상 분할 프레임워크**를 제안한다. 단일 지원 이미지를 증강하고 LPIPS 지표를 통해 쿼리 슬라이스와 최적의 매칭을 수행하여 마스크 프롬프트를 생성하는 방식을 통해, 모델 재학습 없이도 Synapse-CT, CHAOS-MRI, CMR 데이터셋에서 기존 SOTA 성능을 뛰어넘는 성과를 거두었다. 이 연구는 파운데이션 모델을 의료 도메인에 효율적으로 적응시키는 새로운 방향성을 제시하며, 향후 의료 영상 분석의 자동화 및 효율성 제고에 크게 기여할 것으로 기대된다.

# From Dataset to Real-world: General 3D Object Detection via Generalized Cross-domain Few-shot Learning

Shuangzhi Li, Junlong Shen, Lei Ma, and Xingyu Li (2026)

## 🧩 Problem to Solve

본 논문은 LiDAR 기반 3D 객체 검출(Object Detection) 모델이 실제 환경으로 배포되었을 때 겪는 일반화(Generalization) 성능 저하 문제를 해결하고자 한다. 기존 데이터셋들은 특정 도시의 제한된 객체 카테고리(예: 자동차, 보행자, 자전거)에 집중되어 있어, 새로운 지역이나 환경에서 나타나는 새로운 객체 종류(Novel Classes, 예: 태국의 툭툭이나 중국의 전동 스쿠터)를 인식하는 데 어려움이 있다.

특히, 새로운 환경에 적응하기 위해 대규모 LiDAR 데이터를 다시 수집하고 레이블링하는 것은 비용과 시간 측면에서 매우 비효율적이다. 따라서 본 연구의 목표는 **제한된 수의 타겟 도메인 샘플(Few-shot annotations)**만을 사용하여, 기존의 공통 클래스(Common Classes)에 대한 성능을 유지하면서 동시에 새로운 클래스(Novel Classes)를 효과적으로 검출할 수 있는 **Generalized Cross-domain Few-shot (GCFS)** 학습 체계를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **2D Open-set 세만틱(Semantics)과 3D 공간 추론(Spatial Reasoning)을 결합**하여 도메인 간의 격차와 클래스 확장 문제를 동시에 해결하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **GCFS 태스크 정의**: 3D 객체 검출 분야에서 도메인 시프트(Domain Shift)와 새로운 클래스 학습을 동시에 다루는 Generalized Cross-domain Few-shot 학습이라는 새로운 태스크를 공식화하고 그 해결책을 제시하였다.
2.  **Image-guided Multi-modal Fusion (IMMF)**: Vision-Language Model(VLM)을 활용해 이미지에서 추출한 전이 가능한 2D 세만틱 힌트를 3D 파이프라인에 주입하여, 희소한(Sparse) 포인트 클라우드 환경에서도 객체 제안(Proposal)의 품질을 높였다.
3.  **Physically-aware Box Search**: 2D 마스크를 3D 박스로 투영할 때 발생하는 노이즈를 줄이기 위해, LiDAR의 스캐닝 특성을 반영한 물리적 제약 기반의 박스 탐색 전략을 제안하였다.
4.  **Contrastive-enhanced Prototype Learning**: 적은 수의 샘플로부터 변별력 있는 클래스별 세만틱 앵커를 학습하기 위해 대조 학습(Contrastive Learning)을 도입하여 표현 학습의 안정성을 확보하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별성을 강조한다.

-   **3D Few-shot Learning (FSL) & GFSL**: 기존의 3D FSL이나 GFSL 연구들은 학습 환경과 배포 환경의 데이터 분포가 동일하다고 가정한다. 그러나 실제 상황에서는 센서 설정이나 환경적 요인으로 인한 도메인 격차가 존재하므로, 이를 고려하지 않는 기존 방식은 GCFS 설정에서 성능이 낮다.
-   **3D Domain Adaptation (DA)**: 도메인 시프트를 해결하는 데 집중하지만, 학습 과정에서 보지 못한 새로운 객체 카테고리(Novel Classes)를 명시적으로 처리하지 못하며, 대량의 레이블 없는 타겟 데이터가 필요하다는 제약이 있다.
-   **Open-vocabulary 3D Object Detection (OVD)**: VLM을 사용하여 새로운 세만틱을 획득하지만, 대개 많은 양의 타겟 데이터나 시퀀스 데이터를 필요로 하며, 극소수의 샘플만 주어지는 Few-shot 상황에서의 성능은 검증되지 않았다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 프레임워크는 소스 데이터로 사전 학습된 모델 $M_{pretrained}$를 기반으로 하며, 타겟 도메인의 $K$-shot 샘플을 사용하여 $M_{finetuned}$로 정교화하는 과정을 거친다. 전체 파이프라인은 크게 **Image-guided Multi-modal Fusion (IMMF)** 모듈과 **Contrastive-enhanced Prototype Learning** 모듈로 구성된다.

### 2. Image-guided Multi-modal Fusion (IMMF)
LiDAR 데이터의 희소성을 극복하기 위해 2D 이미지의 풍부한 세만틱 정보를 활용한다.
-   **특징 추출 및 융합**: Grounding DINO(GDino)를 통해 2D 박스를 생성하고, SAM을 통해 정밀한 객체 마스크 $M^{2D}$를 얻는다. 이를 3D 공간의 복셀 특징 $F^{voxel}$과 투영 매핑 $f_{proj}$를 통해 결합한다.
-   **융합 방정식**:
    $$F^{fused} = F^{voxel} + \text{MLP}(M^{vxl-obj})$$
    여기서 $M^{vxl-obj}$는 이미지 마스크를 3D 복셀 좌표로 투영한 결과이다.

### 3. Physically-aware Box Search
2D-to-3D 투영 시 발생하는 정렬 오류를 줄이기 위해 물리적 제약을 기반으로 최적의 3D 박스 $[x, y, z, \theta]$를 탐색한다.
-   **객체 구분**: 객체를 단순 구조(Simple Structural, SS; 예: 차량)와 복잡 구조(Complex Structural, CS; 예: 보행자)로 구분하여 서로 다른 손실 함수를 적용한다.
-   **손실 함수**:
    1.  **Outside Distance Loss ($L_{OD}$)**: 포인트들이 박스 경계 내에 위치하도록 강제한다.
    2.  **Front-viewed Distance Loss ($L_{FVD}$)**: SS 객체에 적용하며, LiDAR 스캔 특성상 포인트가 박스의 전면 경계에 집중되는 경향을 이용한다.
    3.  **Bird-viewed Center Loss ($L_{BVC}$)**: CS 객체에 적용하며, BEV(Bird's Eye View) 상에서 포인트의 중심과 박스의 중심을 정렬한다.
-   **최종 박스 손실**: $$L_{box} = L_{OD} + \lambda_1 L_{FVD} + \lambda_2 L_{BVC}$$
    이 최적화는 BFGS 알고리즘을 통해 효율적으로 수행된다.

### 4. Contrastive-enhanced Prototype Learning
제한된 데이터로 인해 발생하는 과적합을 방지하고 클래스 간 변별력을 높이기 위해 학습 가능한 프로토타입 $F^{pro}$를 도입한다.
-   **대조 학습 (Contrastive Learning)**: 타겟 도메인의 Few-shot 샘플 특징 $F^{fs}$를 앵커로 사용하여, 해당 클래스의 프로토타입과는 가깝게, 타 클래스의 프로토타입과는 멀게 학습시키는 InfoNCE 손실을 적용한다.
    $$L_{CL} = -\sum_{c \in C^t} \log \frac{\exp(\text{Sim}(F^{fs}_c, F^{pro}_c)/\tau)}{\sum_{s \in C^t} \exp(\text{Sim}(F^{fs}_c, F^{pro}_s)/\tau)}$$
-   **특징 정교화 (Feature Refinement)**: 추론 시, 제안된 객체 특징 $F^{prp}$를 쿼리(Query)로, 학습된 프로토타입 $F^{pro}$를 키(Key)와 값(Value)으로 하는 Cross-Attention 메커니즘을 통해 특징을 보정한다.
    $$\hat{F}^{prp} = \text{Softmax}\left(\frac{F^{prp} W^Q (F^{pro} W^K)^\top}{\sqrt{d}}\right) F^{pro} W^V$$
    최종 특징은 $\tilde{F}^{prp} = \hat{F}^{prp} + F^{prp}$로 계산되어 검출 헤드로 전달된다.

### 5. 학습 및 추론 절차
-   **최적화**: 전체 손실 함수 $L = L_{rpn} + L_{det} + \lambda L_{CL}$를 통해 모델 파라미터와 프로토타입을 동시에 최적화한다.
-   **Meta-learning**: 빠른 적응을 위해 MAML(Model-Agnostic Meta-Learning) 기반의 학습 체계를 채택하여, 소스 데이터에서 다양한 Few-shot 태스크를 시뮬레이션하며 초기 파라미터를 찾는다.

## 📊 Results

### 1. 실험 설정
-   **벤치마크**: 4가지 GCFS 설정(NuScenes $\rightarrow$ FS-KITTI, Waymo $\rightarrow$ FS-KITTI, KITTI $\rightarrow$ FS-A2D2, KITTI $\rightarrow$ FS-Argo2)을 구축하였다.
-   **데이터**: 각 클래스당 $K=5$개의 샘플을 기본으로 사용하였으며, $K \in \{1, 3, 5, 10, 20, 40\}$의 변화에 따른 성능을 분석하였다.
-   **지표**: mAP (mean Average Precision)를 주요 지표로 사용하였다.
-   **비교 대상**: Target-FT(단순 미세 조정), Proto-Vote, PVAE-Vote, CP-Vote(3D-FSL 방식), GFS-Det(3D-GFSL 방식) 등과 비교하였다.

### 2. 주요 결과
-   **정량적 성능**: 모든 벤치마크에서 제안 방법이 기존 SOTA 모델들을 압도하였다. 특히 NuScenes $\rightarrow$ FS-KITTI 설정에서 전체 mAP 13.55%를 기록하여 Baseline(Target-FT, 8.61%) 대비 큰 향상을 보였다.
-   **클래스별 성능**: 공통 클래스(Common)뿐만 아니라 새로운 클래스(Novel)에서도 강력한 성능을 보였다. 이는 VLM 기반의 이미지 융합이 새로운 객체의 Recall을 크게 높였기 때문으로 분석된다.
-   **K-shot 확장성**: $K$값이 증가함에 따라 성능이 꾸준히 향상되었으며, $K=40$일 때는 Full-shot 학습 결과에 근접하는 성능을 보였다.

### 3. Ablation Study 결과
-   **컴포넌트 분석**: Image-Fusion은 특히 Novel 클래스의 성능을 크게 높였고, CL-Proto는 Common 클래스의 변별력을 높여 전체 성능을 상호 보완적으로 향상시켰다.
-   **박스 탐색**: $L_{OD}$만 사용했을 때보다 $L_{FVD}$와 $L_{BVC}$를 함께 사용했을 때, 객체의 구조적 복잡성에 관계없이 더 정확한 박스 제안이 가능함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점
-   **멀티모달 시너지**: 2D VLM의 풍부한 세만틱 지식과 3D의 기하학적 구조 정보를 효과적으로 결합하여, 데이터가 극도로 부족한 상황에서도 강건한 일반화 성능을 확보하였다.
-   **도메인 유연성**: 다양한 빔(Beam) 수(16, 32, 64-beam)를 가진 LiDAR 센서 간의 도메인 시프트 상황에서도 안정적인 성능을 유지하며 높은 전이 가능성을 입증하였다.

### 2. 한계 및 논의사항
-   **난이도 높은 클래스**: "Person sitting"과 같이 구조가 모호하고 다양한 클래스에 대해서는 여전히 성능 향상 폭이 낮았다. 이는 단순한 세만틱 힌트만으로는 해결하기 어려운 기하학적 모호성이 존재함을 시사한다.
-   **시프트가 적은 환경**: 도메인 시프트가 거의 없는 환경(예: Waymo $\rightarrow$ FS-KITTI의 공통 클래스)에서는 새로운 클래스 학습으로 인한 간섭으로 인해 공통 클래스의 성능 향상이 미미한 경우가 관찰되었다.

### 3. 비판적 해석
본 연구는 VLM을 통해 2D-to-3D의 가교를 성공적으로 구축하였으나, 추론 속도 측면에서 VLM과 SAM의 연산 비용이 발생한다는 점이 실제 실시간 시스템 적용 시 병목이 될 수 있다. 다만, 논문에서 언급한 mixed-precision 최적화를 통해 10.11 FPS를 달성한 점은 고무적이다.

## 📌 TL;DR

본 논문은 3D 객체 검출에서 **도메인 시프트와 새로운 클래스 학습을 동시에 해결하는 Generalized Cross-domain Few-shot (GCFS)** 태스크를 제안하고, 이를 해결하기 위해 **VLM 기반의 이미지-3D 융합(IMMF)**과 **대조 학습 기반의 프로토타입 학습**을 결합한 프레임워크를 제시하였다. 실험 결과, 극소수의 샘플만으로도 새로운 환경의 객체를 효과적으로 검출할 수 있음을 입증하였으며, 이는 향후 자율주행 시스템이 새로운 도시나 환경에 빠르게 적응하는 데 핵심적인 역할을 할 것으로 기대된다.
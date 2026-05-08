# Video Instance Segmentation with a Propose-Reduce Paradigm

Huaijia Lin, Ruizheng Wu, Shu Liu, Jiangbo Lu, Jiaya Jia

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 작업에서 발생하는 기존 방법론의 한계를 해결하고자 한다. VIS는 비디오의 각 프레임에서 미리 정의된 클래스의 모든 인스턴스를 분할하고, 이를 전체 비디오에 걸쳐 동일한 인스턴스로 연결(associate)하는 것을 목표로 한다.

기존의 VIS 접근 방식은 크게 'Track-by-Detect'와 'Clip-Match' 두 가지 패러다임으로 나뉜다. 전자는 각 프레임별로 검출 및 분할을 수행한 후 프레임 간 추적(tracking)을 통해 연결하며, 후자는 비디오를 짧은 클립으로 나누어 처리한 후 클립 간 매칭(matching)을 통해 연결한다. 이러한 방식들은 공통적으로 불완전한 결과물(단일 프레임 또는 짧은 클립)을 생성한 뒤 이를 병합하는 단계를 거치는데, 이 과정에서 특히 폐색(occlusion)이나 빠른 움직임이 발생할 때 오류가 누적(error accumulation)되는 치명적인 문제가 발생한다.

따라서 본 논문의 목표는 불완전한 시퀀스를 병합하는 단계 없이, 단 한 번의 단계로 비디오 전체에 대한 완전한 인스턴스 시퀀스를 생성함으로써 오류 누적 문제를 해결하고 강건성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **'Propose-Reduce'**라는 새로운 패러다임을 제안한 것이다. 이는 이미지 레벨의 객체 검출(Object Detection)에서 사용되는 Two-stage 프레임워크(RPN으로 후보군 생성 후 NMS로 제거)의 직관을 비디오 도메인으로 확장한 것이다.

1. **Propose-Reduce 패러다임 제안**: 여러 개의 키 프레임(Key frames)으로부터 인스턴스 시퀀스 후보(Sequence proposals)를 다수 생성하고, 이후 동일 인스턴스에 대해 중복 생성된 시퀀스들을 제거(Reduce)하여 최종 결과를 얻는 방식이다.
2. **Seq Mask R-CNN 설계**: Mask R-CNN을 기반으로 하며, 장기 전파(long-term propagation)를 위해 **Sequence Propagation Head (Seq-Prop head)**를 추가하여 이미지 레벨 네트워크를 비디오 도메인으로 확장하였다.
3. **성능 입증**: YouTube-VIS와 DAVIS-UVOS 벤치마크 데이터셋에서 State-of-the-art (SOTA) 성능을 달성하였다.

## 📎 Related Works

### 이미지 레벨 인스턴스 분할 (Image-Level Instance Segmentation)

Mask R-CNN과 같은 Top-down 방식이 높은 성능으로 인해 널리 사용되고 있다. 본 연구는 이러한 Mask R-CNN의 구조를 기본 뼈대로 채택하여 비디오 도메인으로 확장하였다.

### 비디오 인스턴스 분할 (Video Instance Segmentation)

기존 연구들은 'Track-by-Detect'와 'Clip-Match' 방식으로 구분된다. 하지만 앞서 언급했듯이, 이들은 모두 개별 프레임이나 클립의 결과를 병합하는 과정에서 오류가 누적되는 한계가 있다.

### 반지도/비지도 비디오 객체 분할 (Semi/Unsupervised VOS)

STM(Space-Time Memory Networks)과 같은 메모리 기반 전파(propagation) 방식이 장기 전파 문제를 해결하며 주목받았다. 본 논문의 Seq-Prop head는 STM의 전파 전략에서 영감을 얻었으나, 별도의 백본을 사용하는 STM과 달리 Mask R-CNN의 백본을 공유함으로써 연산 효율성을 높였다.

## 🛠️ Methodology

### 전체 파이프라인

Propose-Reduce 패러다임은 크게 두 단계로 구성된다.

1. **Sequence Proposal Generation**: 다수의 키 프레임을 선정하고, 각 키 프레임에서 검출된 인스턴스를 비디오 전체로 전파하여 다수의 시퀀스 후보군을 생성한다.
2. **Sequence Proposal Reduction**: 생성된 후보군 중 중복되는 시퀀스를 제거하여 최종 결과셋을 도출한다.

### 1. 시퀀스 후보 생성 (Sequence Proposals Generation)

#### 키 프레임 선정 (Key Frames Selection)

전체 $T$ 프레임 비디오에서 $K$개의 키 프레임을 균등한 간격으로 선정한다. 키 프레임의 인덱스 $g(k)$는 다음과 같이 결정된다.
$$g(k) = \max\{\lfloor T/K \rfloor, 1\} \times k, \quad k= 0, \dots, K-1$$

#### Memory K-Propagation

선정된 $K$개의 키 프레임 각각에 대해 이미지 레벨 인스턴스 분할을 수행하고, 이를 양방향(bi-directionally)으로 전파하여 $K$개의 마스크 시퀀스 집합 $\{S_0, S_1, \dots, S_{K-1}\}$을 생성한다. 이때 STM에서 사용된 메모리(Memory) 개념을 도입하여 이전 프레임들의 인코딩된 특징을 저장하고 이를 현재 프레임의 전파에 활용함으로써 오류 누적을 완화한다.

#### Seq Mask R-CNN 아키텍처 및 학습

Seq Mask R-CNN은 Mask R-CNN에 **Seq-Prop head**를 추가한 구조이다.

- **작동 원리**: 가이드 프레임(Guidance frame, $t$)의 추정 마스크 $M_g$와 FPN 특징 $P_g^2$를 입력받아 인코딩 특징 $F_g$를 생성한다. 쿼리 프레임(Query frame, $t+\delta$)의 FPN 특징 $P_q^2$로부터 $F_q$를 생성한 뒤, **Non-local (NL) operation**을 통해 가이드 프레임의 마스크 정보를 쿼리 프레임으로 전파하여 $F_{g \to q}$를 얻는다. 최종적으로 쿼리 프레임의 백본 특징 $C_q^2$와 결합하여 쿼리 마스크 $M_q$를 생성한다.
- **학습 목표 및 손실 함수**:
    전체 손실 함수 $L$은 다음과 같이 다중 작업 손실(multi-task loss)로 정의된다.
    $$L = L_{cls} + L_{box} + L_{mask} + L_{prop}$$
    여기서 $L_{cls}, L_{box}, L_{mask}$는 Mask R-CNN의 기존 손실 함수와 동일하며, 전파 학습을 위한 $L_{prop}$는 다중 스케일 인스턴스를 동시에 처리하기 위해 **scale-balanced soft IoU loss**를 사용한다.

### 2. 시퀀스 후보 제거 (Sequence Proposals Reduction)

서로 다른 키 프레임에서 동일한 인스턴스가 중복 검출될 수 있으므로, 시퀀스 레벨의 NMS(Non-Maximum Suppression) 변형 알고리즘을 적용한다.

- **시퀀스 점수 (Sequence Score)**: 각 인스턴스 시퀀스의 우선순위를 결정하기 위해 모든 프레임의 분류 점수를 평균 낸 뒤, 클래스 중 최대값을 취한다.
    $$C(S_k^o) = \max_{|C|} \frac{1}{T} \sum_{t=0}^{T-1} C(S_k^o(t))$$
- **시퀀스 IoU (Sequences IoU)**: 두 시퀀스 간의 겹침 정도를 측정하기 위해 모든 프레임의 마스크 IoU 합계를 계산한다.
    $$\text{IoU}(S_k^o, S_{\tilde{k}}^{\tilde{o}}) = \frac{\sum_{t=0}^{T-1} |M(S_k^o(t)) \cap M(S_{\tilde{k}}^{\tilde{o}}(t))|}{\sum_{t=0}^{T-1} |M(S_k^o(t)) \cup M(S_{\tilde{k}}^{\tilde{o}}(t))|}$$
이후 정의된 점수와 IoU를 바탕으로 전통적인 NMS 알고리즘을 적용하여 중복 시퀀스를 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS (40개 카테고리), DAVIS-UVOS (전경/배경 2개 카테고리).
- **지표**: YouTube-VIS는 AP 및 AR를 사용하며, DAVIS-UVOS는 J-score(Mean IoU)와 F-score(경계 정확도)의 평균인 J&F score를 사용한다.
- **구현**: COCO 데이터셋의 이미지를 회전시켜 생성한 가상 비디오를 통해 사전 학습(main-training) 후, 각 비디오 데이터셋으로 미세 조정(finetuning)하는 2단계 학습 전략을 취했다.

### 주요 결과

1. **정량적 결과**:
    - **YouTube-VIS**: ResNeXt-101 백본 기준 **47.6% AP**를 달성하여 MaskProp(46.6%) 등 기존 SOTA 모델을 능가하였다. 특히 AR@10 지표에서 큰 향상을 보여, 다중 키 프레임 샘플링 전략이 리콜(Recall) 성능을 높였음을 입증하였다.
    - **DAVIS-UVOS**: ResNeXt-101 백본 기준 **70.4% J&F score**를 기록하며 UnOVOST 등의 복잡한 멀티-모델 시스템보다 우수한 성능을 보였다.
2. **정성적 결과**:
    - 긴 시간 동안 폐색이 발생하는 시나리오에서 'Track-by-Detect'나 'Clip-Match' 방식은 객체를 놓치거나 서로 다른 인스턴스로 오인하는 반면, 제안 방법은 장기 전파를 통해 완전한 시퀀스를 성공적으로 생성하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 VIS의 문제를 '병합'의 관점이 아닌 '제안 및 제거(Propose-Reduce)'의 관점으로 재정의함으로써, 병합 단계에서 발생하는 오류 누적 문제를 원천적으로 차단하였다. 또한, 이미지 레벨의 강력한 모델인 Mask R-CNN에 가벼운 Seq-Prop head를 추가하는 단순한 구조만으로도 매우 높은 성능을 낼 수 있음을 보여주었다.

### 한계 및 비판적 해석

- **동일 카테고리 중복**: 시각화 결과에서 동일한 카테고리의 인스턴스들이 서로 겹쳐 있을 때(예: 여러 명의 사람이 겹쳐 있는 경우), 한 사람의 팔이 다른 사람의 마스크로 전파되는 오류가 관찰되었다. 이는 모델이 인스턴스 간의 세밀한 구분보다는 카테고리 특징에 의존하여 전파하는 경향이 있음을 시사한다.
- **하이퍼파라미터 $K$의 영향**: 키 프레임의 수 $K$가 너무 적으면 리콜이 낮아지고, 너무 많으면 거짓 양성(False Positive)이 증가하여 AP가 하락하는 트레이드-오프 관계가 존재한다. 이는 $K$값 선정에 있어 데이터셋별 최적화가 필요함을 의미한다.

## 📌 TL;DR

본 논문은 비디오 인스턴스 분할(VIS)에서 기존의 병합 기반 방식이 갖는 오류 누적 문제를 해결하기 위해, **다수의 키 프레임으로부터 시퀀스 후보를 생성하고 중복을 제거하는 'Propose-Reduce' 패러다임**을 제안한다. 이를 위해 Mask R-CNN에 전파 헤드를 추가한 **Seq Mask R-CNN**을 설계하였으며, YouTube-VIS와 DAVIS-UVOS 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 복잡한 추적/매칭 모듈 없이도 장기 전파와 시퀀스 레벨의 NMS만으로 강건한 VIS가 가능함을 입증하였으며, 향후 이미지 레벨의 고성능 모델을 비디오 도메인으로 확장하는 효율적인 방향성을 제시한다.

# Metric Learning for Keyword Spotting

Jaesung Huh, Minjae Lee, Heesoo Heo, Seongkyu Mun, Joon Son Chung (2020)

## 🧩 Problem to Solve

본 논문은 키워드 스포팅(Keyword Spotting, KWS) 시스템에서 발생하는 높은 오경보율(False Alarm rate) 문제를 해결하고자 한다. 기존의 KWS 연구들은 대부분 타겟 키워드와 비타겟(non-target) 키워드가 미리 정의된 폐쇄 집합 분류(closed-set classification) 문제로 접근해 왔다. 이러한 방식은 학습 과정에서 보지 못한(unseen) 다양한 소리들이 입력되었을 때, 이를 적절히 거르지 못하고 타겟 키워드로 잘못 분류하는 문제가 발생한다.

실제 환경에서 KWS는 미리 정의된 타겟 키워드를 수많은 알 수 없는 소리들 사이에서 찾아내는 '검출(detection)' 문제에 가깝다. 따라서 본 연구의 목표는 Metric Learning을 통해 타겟 키워드와 비타겟 소리 간의 거리를 극대화함으로써, 학습 시 경험하지 못한 비타겟 소리에 대한 일반화 성능을 높이고 오경보율을 낮추는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 KWS를 단순 분류 문제가 아닌 검출 문제로 재정의하고, Metric Learning을 도입하여 타겟 클래스의 응집도는 높이되 비타겟 클래스는 하나의 점으로 모으지 않는 전략을 취하는 것이다. 주요 기여 사항은 다음과 같다.

1. **문제 정의의 전환**: KWS를 폐쇄 집합 분류에서 오픈 세트 검출 문제로 재구성하여, 학습되지 않은 비타겟 소리에 대한 대응력을 높였다.
2. **Metric Learning 기반 학습 전략**: Triplet Loss와 Prototypical Loss를 KWS에 적용하였으며, 특히 비타겟 소리들을 하나의 클러스터로 묶지 않는 방식을 통해 실제 환경의 높은 변동성을 반영하였다.
3. **AP-FC(Angular Prototypical with Fixed Classes) 제안**: 타겟 키워드가 고정되어 있다는 특성을 이용해, 동적인 Centroid 대신 학습 가능한 가중치(per-class weights)를 사용하는 개선된 Prototypical 네트워크를 제안하였다.
4. **SVM 기반 추론 방법**: Metric Learning으로 학습된 임베딩 공간에서 효과적인 결정을 내리기 위해 One-vs-Rest RBF 커널 SVM을 도입하였다.

## 📎 Related Works

기존의 KWS 연구들은 주로 CNN 기반의 분류기(Classifier)를 사용하여 타겟 키워드와 비타겟 소리(일반 음성 및 소음)를 구분하였다. 하지만 이러한 방식은 비타겟 소리의 다양성을 충분히 반영하지 못하며, 한정된 수의 비타겟 클래스만을 사용하여 학습하기 때문에 실제 환경에서의 일반화 성능이 떨어진다는 한계가 있다.

Metric Learning 분야에서는 Contrastive Loss나 Triplet Loss 등이 얼굴 인식 및 화자 검증(Speaker Verification) 등에 널리 사용되어 왔다. 최근에는 페어 선택(pair selection)의 어려움을 해결하기 위해 Prototypical Networks와 같은 접근 방식이 등장하였다. 본 논문은 이러한 Metric Learning의 직관을 KWS에 접목하되, 타겟 키워드가 미리 정해져 있다는 점을 활용하여 기존의 화자 검증 방식과 차별화를 두었다.

## 🛠️ Methodology

### 1. 기본 손실 함수 (Loss Functions)

본 연구에서는 두 가지 Metric Learning 손실 함수를 사용한다.

**Triplet Loss**
동일 클래스 간의 거리는 좁히고, 서로 다른 클래스 간의 거리는 넓히는 방식이다.
$$L = \sum_{i} \max (0, \|f(x_i) - f(x'_i)\| - \|f(x_i) - f(x_j)\| + \alpha)$$
여기서 $f(x)$는 임베딩 벡터이며, $x_i, x'_i$는 동일 클래스, $x_j$는 다른 클래스의 샘플이다. $\alpha$는 마진(margin) 값이다.

**Angular Prototypical Loss**
각 클래스의 중심(Centroid)과 임베딩 간의 코사인 유사도를 기반으로 학습한다.
유사도 $S_{j,k}$는 다음과 같이 정의된다.
$$S_{j,k} = w \cdot \cos(e_{j,M}, c_k) + b$$
여기서 $c_k$는 $k$번째 클래스의 중심 벡터이며, $w$와 $b$는 학습 가능한 파라미터이다. 최종 손실 함수는 소프트맥스 형태의 교차 엔트로피를 따른다.
$$L = -\frac{1}{N} \sum_{j=1}^{N} \log \frac{e^{S_{j,j}}}{\sum_{k=1}^{N} e^{S_{j,k}}}$$

### 2. 학습 전략 및 추론 절차

본 논문은 세 가지 전략을 비교 분석한다.

*   **Metric Learning with an unknown cluster**: 타겟과 비타겟 모두를 각각의 클러스터로 묶는 기본 방식이다. 추론 시에는 각 클래스의 Centroid와 테스트 샘플 간의 유사도를 측정하여 분류한다.
*   **Metric Learning without an unknown cluster**: 비타겟 소리들의 변동성이 매우 크다는 점에 착안하여, 비타겟 샘플들을 하나의 점으로 모으지 않는다. 즉, 타겟 클래스에 대해서만 내적 변동성을 최소화하도록 손실 함수를 수정한다. 추론 시에는 Centroid를 사용할 수 없으므로, 학습된 임베딩을 이용하여 **One-vs-Rest RBF 커널 SVM**을 학습시켜 최종 판단을 내린다.
*   **AP-FC (Angular Prototypical with Fixed Classes)**: 타겟 클래스가 고정되어 있다는 점을 활용해, Centroid $c_k$를 매번 계산하는 대신 학습 가능한 가중치 $W_k$로 대체한다.
    $$S_{j,i,k} = w \cdot \cos(e_{j,i}, W_k) + b, \quad k \in \{\text{target}\}$$
    이를 통해 분류기의 높은 타겟 정확도와 Metric Learning의 비타겟 거부 능력을 동시에 확보한다.

### 3. 시스템 아키텍처 및 학습 설정

*   **입력 데이터**: 16kHz 오디오 신호를 40차원 MFCC(Mel-Frequency Cepstrum Coefficient)로 변환하여 사용하며, 모든 데이터는 1초 길이로 고정한다.
*   **백본 네트워크**: `res15` 아키텍처를 사용한다. 이는 Residual Connection과 Dilated Convolution을 포함하여 수용 영역(receptive field)을 넓힌 구조이다. 최종 층은 $D$차원(예: 32)의 임베딩 벡터를 출력하는 FC 레이어로 구성된다.
*   **데이터 증강**: 학습 시 20%의 입력 피처를 시간 축으로 $\pm 200\text{ms}$ 범위 내에서 랜덤하게 시프트(shift)하고 제로 패딩을 적용한다.

## 📊 Results

### 실험 설정
*   **데이터셋**: Google Speech Commands v0.01을 사용한다.
*   **데이터 분할 (Custom Split)**: 실제 환경을 모사하기 위해 20개의 비타겟 키워드 중 10개는 학습에 사용(seen-unknown)하고, 나머지 10개는 테스트에만 사용(unseen-unknown)하도록 분리하였다.
*   **평가 지표**: 전체 정확도(Total Acc), 타겟 정확도(Target Acc), 비타겟 정확도(Non-tgt Acc), AUC, mAP를 측정하였다. 특히 타겟과 비타겟의 비율을 11:1과 1:1(실제 환경과 유사) 두 가지 설정에서 평가하였다.

### 정량적 결과
| 방법론 | Back-end | Total Acc (1:1) | Target Acc | Non-tgt Acc | mAP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline (CE) | Softmax | 73.81% | 95.52% | 52.11% | 94.84% |
| Triplet | Centroid | 76.59% | 93.45% | 59.47% | 92.80% |
| AP | Centroid | 77.80% | 95.40% | 60.22% | 95.19% |
| Triplet | SVM | 78.96% | 92.68% | 65.24% | 92.05% |
| AP | SVM | 76.40% | 94.49% | 58.30% | 94.18% |
| **AP-FC** | **SVM** | **83.82%** | **94.26%** | **73.37%** | **95.42%** |

### 결과 분석
1. **비타겟 거부 능력 향상**: 모든 Metric Learning 기반 방법이 Baseline(Cross Entropy)보다 비타겟 정확도(Non-tgt Acc)가 월등히 높다. 이는 학습되지 않은 소리를 거르는 능력이 향상되었음을 의미한다.
2. **전략의 유효성**: 비타겟을 클러스터링하지 않고 SVM을 사용하는 방식이 일반적인 Metric Learning보다 성능이 좋았다.
3. **AP-FC의 우수성**: AP-FC SVM 조합이 전체 정확도, 비타겟 정확도, mAP 모든 지표에서 가장 우수한 성능을 보였으며, 특히 실제 환경과 유사한 1:1 비율 설정에서 매우 강력한 성능을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 KWS 시스템의 고질적인 문제인 '알 수 없는 소리에 대한 오작동'을 Metric Learning의 관점에서 효과적으로 해결하였다. 

**강점 및 시사점**
*   비타겟 데이터를 하나의 클래스로 묶어 강제로 수렴시키려는 기존의 분류 방식이 오히려 일반화 성능을 해친다는 점을 밝혀냈다.
*   타겟 클래스의 고정된 특성을 활용한 AP-FC 구조는 분류기의 정밀함과 Metric Learning의 유연성을 동시에 잡은 효율적인 설계이다.

**한계 및 논의사항**
*   Metric Learning 방법론 중 일부(예: Triplet)는 타겟 키워드 자체의 정확도가 Baseline보다 약간 낮아지는 경향이 있는데, 이는 타겟 내 응집도보다 클래스 간 거리 확보에 더 치중하기 때문으로 분석된다.
*   SVM 백엔드를 사용할 경우, 추론 단계에서 추가적인 계산 비용이나 메모리 사용량이 발생할 수 있으며 이에 대한 실시간 최적화 논의는 본문에서 구체적으로 다루어지지 않았다.

## 📌 TL;DR

본 연구는 키워드 스포팅(KWS)을 단순 분류가 아닌 '검출' 문제로 정의하고, Metric Learning을 적용하여 학습되지 않은 소리에 대한 오경보율을 획기적으로 낮추었다. 특히 타겟 클래스의 가중치를 고정하여 학습하는 **AP-FC** 방법과 **SVM** 기반 추론을 결합하여, 타겟 인식률을 유지하면서도 비타겟 소리 거부 능력을 크게 향상시켰다. 이 결과는 실제 환경처럼 예측 불가능한 소음이 많은 상황에서 KWS 시스템의 신뢰성을 높이는 데 중요한 기여를 할 것으로 보인다.
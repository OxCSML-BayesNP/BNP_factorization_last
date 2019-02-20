
"""
  Computes the area under the ROC curve

  Parameters:
    rates: (array) Estimated probability of links
    links: (array) Observed links
"""
function auc_roc(rate::Array{Float64,1},
                links::Array{Int64,1})
  sorted_idx = sortperm(rate)
  sorted_links = links[sorted_idx]
  count_links = countnz(links)
  count_non_links = countnz(1-links)
  #return sorted_links, count_links, count_non_links
  return (2*sum(find(sorted_links)) - count_links*(count_links+1))/(2*count_non_links *count_links)
end



"""
Copyright 2016 Tamas Nagy, Martin Kampmann, and contributers

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a
copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
"""

function Base.count(labels::AbstractArray{Symbol}, pos_labels::Set{Symbol})
    num_pos, num_neg = 0, 0
    for label in labels
        if label in pos_labels
            num_pos += 1
        else
            num_neg += 1
        end
    end
    num_pos, num_neg
end

"""
auprc(scores::AbstractArray{Float64}, classes::AbstractArray{Symbol}, pos_labels::Set{Symbol})

Computes the area under the Precision-Recall curve using a lower
trapezoidal estimator, which is more accurate for skewed datasets.
"""
function auprc(scores::AbstractArray{Float64}, classes::AbstractArray{Symbol}, pos_labels::Set{Symbol})
    num_scores = length(scores) + 1
    ordering = sortperm(scores, rev=true)
    labels = classes[ordering]
    num_pos, num_neg = count(labels, pos_labels)

    tn, fn, tp, fp = 0, 0, num_pos, num_neg

    p = Array(Float64, num_scores)
    r = Array(Float64, num_scores)
    p[num_scores] = tp/(tp+fp)
    r[num_scores] = tp/(tp+fn)
    auprc, prev_r = 0.0, r[num_scores]
    pmin, pmax = p[num_scores], p[num_scores]

    # traverse scores from lowest to highest
    for i in num_scores-1:-1:1
        dtn = labels[i] in pos_labels ? 0 : 1
        tn += dtn
        fn += 1-dtn
        tp = num_pos - fn
        fp = num_neg - tn
        p[i] = (tp+fp) == 0 ? 1-dtn : tp/(tp+fp)
        r[i] = tp/(tp+fn)

        # update max precision observed for current recall value
        if r[i] == prev_r
            pmax = p[i]
        else
            pmin = p[i] # min precision is always at recall switch
            auprc += (pmin + pmax)/2*(prev_r - r[i])
            prev_r = r[i]
            pmax = p[i]
        end
    end
    auprc, p, r
end


"""
auc_pr(scores::AbstractArray{Float64}, classes::AbstractArray{Symbol}, pos_labels::Set{Symbol})

Computes the area under the Precision-Recall curve using a lower
trapezoidal estimator, which is more accurate for skewed datasets.
"""
function auc_pr(scores::Array{Float64,1}, classes::Array{Int64,1})
    num_scores = length(scores) + 1
    ordering = sortperm(scores, rev=true)
    labels = classes[ordering]
    num_pos = sum(classes)
    num_neg = length(classes) - num_pos

    tn, fn, tp, fp = 0, 0, num_pos, num_neg

    p = Array(Float64, num_scores)
    r = Array(Float64, num_scores)
    p[num_scores] = tp/(tp+fp)
    r[num_scores] = tp/(tp+fn)
    auprc, prev_r = 0.0, r[num_scores]
    pmin, pmax = p[num_scores], p[num_scores]

    # traverse scores from lowest to highest
    for i in num_scores-1:-1:1
        dtn = labels[i] == 1 ? 0 : 1
        tn += dtn
        fn += 1-dtn
        tp = num_pos - fn
        fp = num_neg - tn
        p[i] = (tp+fp) == 0 ? 1-dtn : tp/(tp+fp)
        r[i] = tp/(tp+fn)

        # update max precision observed for current recall value
        if r[i] == prev_r
            pmax = p[i]
        else
            pmin = p[i] # min precision is always at recall switch
            auprc += (pmin + pmax)/2*(prev_r - r[i])
            prev_r = r[i]
            pmax = p[i]
        end
    end
    auprc
end

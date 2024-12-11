package tienthuan.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Date;

@Data
@Entity
@Builder
@Table(name = "history")
@NoArgsConstructor
@AllArgsConstructor
public class History {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    @Column(name = "accept")
    private Boolean accept;

    @Column(name = "average_occurrence_score")
    private Double averageOccurrenceScore;

    @Column(name = "average_similarity_score")
    private Double averageSimilarityScore;

    @Column(name = "most_common_id")
    private String mostCommonId;

    @Column(name = "occurrence_count")
    private Integer occurrenceCount;

    @Column(name = "score")
    private Double score;

    @Column(name = "history_date")
    private Date historyDate;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
}

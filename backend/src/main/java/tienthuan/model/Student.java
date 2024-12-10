package tienthuan.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Collection;

@Data
@Entity
@Builder
@Table(name = "students")
@NoArgsConstructor
@AllArgsConstructor
public class Student {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    @Column(name = "student_code")
    private String studentCode;

    @Column(name = "student_name")
    private String studentName;

    @Column(name = "is_registered")
    private Boolean isRegistered;

    @OneToMany(mappedBy = "student", fetch = FetchType.EAGER)
    private Collection<PalmPrintImage> palmPrintImages;

}

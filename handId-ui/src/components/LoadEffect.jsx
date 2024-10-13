

export default function LoadEffect() {
    return (
        <div style={styles.spinnerContainer}>
            <div className="spinner-border text-primary" style={styles.spinner} role="status">
                <span className="visually-hidden">Loading...</span>
            </div>
        </div>
    )
}

const styles = {
    spinnerContainer: {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",  
    },
    spinner: {
        width: "5rem",  
        height: "5rem",
    },
};

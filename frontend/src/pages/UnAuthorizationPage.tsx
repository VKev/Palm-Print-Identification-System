
export default function UnAuthorizationPage() {
    return (
        <section className="bg-white dark:bg-gray-900">
            <div className="py-8 px-4 mx-auto max-w-screen-xl lg:py-16 lg:px-6">
                <div className="mx-auto max-w-screen-sm text-center">
                    <h1 className="mb-5 text-7xl tracking-tight font-extrabold lg:text-9xl text-blue-600">403</h1>
                    <p className="mb-5 text-3xl tracking-tight font-bold text-gray-900 md:text-4xl dark:text-white">You do not have access to the path.</p>
                    <p className="mb-8 text-lg font-light text-gray-500 dark:text-gray-400">Sorry, you can't access that page. You must login to acccess the website. </p>
                    <a href="/login" className="text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2">
                        Back to Login page
                    </a>
                </div>
            </div>
        </section>
    )
}

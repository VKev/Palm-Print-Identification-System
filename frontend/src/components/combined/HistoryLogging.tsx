import { useEffect, useState } from 'react';
import useAxios from '../../utils/useAxios';
import { UserProfile } from '../../models/User';
import API from '../../config/API';
import HttpStatus from '../../config/HttpStatus';

interface HistoryResponse {
    id: number;
    accept: boolean;
    averageOccurrenceScore: number;
    averageSimilarityScore: number;
    mostCommonId: string;
    occurrenceCount: number;
    score: number;
    historyDate: string;
}

const ITEMS_PER_PAGE = 10;

export default function HistoryLogging({ user }: { user: UserProfile | null }) {

    const api = useAxios();
    const [historylogs, setHistoryLogs] = useState<HistoryResponse[]>([]);
    const [currentPage, setCurrentPage] = useState(1);

    useEffect(() => {
        const fetchData = async () => {
            const response = await api.get(API.Staff.GET_HISTORY_LOGS_BY_STAFF + user?.id);
            if (response.status === HttpStatus.OK) {
                setHistoryLogs(response.data);
            }
        }
        fetchData().catch(console.error);
    }, [])

    const paginatedData = historylogs.slice(
        (currentPage - 1) * ITEMS_PER_PAGE,
        currentPage * ITEMS_PER_PAGE
    );

    const totalPages = Math.ceil(historylogs.length / ITEMS_PER_PAGE);

    return (
        <div className="container mx-auto">
            <div className="mt-3 text-4xl text-center font-medium">History Logs</div>
            <hr className="mt-5 mb-2" />
            <div className="overflow-x-auto">
                <table className="min-w-full bg-white border border-gray-200">
                    <thead>
                        <tr>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">ID</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">STATUS</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Avg Occurrence Score</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Avg Similarity Score</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Most Common ID</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Occurrence Count</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Score</th>
                            <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">History Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedData.map((item) => (
                            <tr key={item.id} className="hover:bg-gray-100">
                                <td className="py-4 px-6 text-sm font-medium text-gray-900 text-center">{item.id}</td>
                                <td>
                                    <span className={`py-2 px-4 text-sm font-medium rounded-full text-center ${item.accept ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'}`}>
                                        {item.accept ? '✅ Accept' : '❌ Reject'}
                                    </span>
                                </td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">{item.averageOccurrenceScore.toFixed(10)}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">{item.averageSimilarityScore.toFixed(10)}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">{item.mostCommonId}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">{item.occurrenceCount}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">{item.score.toFixed(10)}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 text-center">
                                    {new Date(item.historyDate).toLocaleString('en-GB', {
                                        day: '2-digit',
                                        month: '2-digit',
                                        year: 'numeric',
                                        hour: '2-digit',
                                        minute: '2-digit',
                                        hour12: false
                                    })}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="flex justify-center mt-4">
                <button
                    className='text-sm text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'
                    onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                >
                    Previous
                </button>
                <span className="px-4 py-2 mx-1">{currentPage} / {totalPages}</span>
                <button
                    className='text-sm text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'
                    onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                >
                    Next
                </button>
            </div>
        </div>
    );
}
